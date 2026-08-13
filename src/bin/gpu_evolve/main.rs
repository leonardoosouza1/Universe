//! Evolução genética de verdade dentro do laço todo-GPU.
//!
//! # O que este binário faz
//!
//! Roda ciclos de: N ticks (GPU) → avalia → seleciona → muta → repete.
//! A avaliação lê a posição final da física (CPU side) para calcular
//! distância acumulada com odômetro contínuo.
//!
//! # Fitness SEM DEGRAU
//!
//! Usa distância CONTÍNUA em metros (integral de |vel|×dt por tick).
//! NÃO usa contagem inteira de voltas — medido no passado: ir de 3,01 para
//! 3,99 voltas rendia fitness idêntica, cegando a seleção na faixa crítica.
//!
//! # População inicial
//!
//! Aleatória, gerada por `Genome::random(seed=índice)` com pesos N(0, σ=0.30).
//! Pesos zerados (BitLinear::new padrão) → rede nula → nenhum movimento.
//! σ=0.30 garante pelo menos ~30% de pesos com magnitude suficiente pra acionar
//! o cérebro. NÃO carrega CHAMPION.dna.
//!
//! # Persistência
//!
//! A cada geração salva o melhor genoma em `EVOLVE_BEST_GEN<N>.bin`.
//! Ao iniciar com `--retomar`, tenta carregar `EVOLVE_BEST_GEN*.bin` mais
//! recente e re-semente a população a partir dele.
//!
//! # Relatório
//!
//! Por geração: ≥1V, ≥2V, mediana, máximo, mortalidade.
//! Ao final: inclinação (OLS) com σ e teste de janela encolhida
//! (remove últimas 1, 2, 3 gens — se inclinação some, era artefato de janela).
//!
//! # Uso
//!
//!   cd ~/projects/LLM/Universe
//!   env -u DISPLAY nice -n 19 ./target/release/gpu_evolve \
//!       --seres 60 --ticks 8000 --gens 100
//!
//!   # Para retomar de onde parou:
//!   env -u DISPLAY nice -n 19 ./target/release/gpu_evolve \
//!       --seres 60 --ticks 8000 --gens 100 --retomar

use std::time::Instant;
use std::sync::atomic::Ordering;

mod statics;
mod types;
mod genome_io;
mod audit;
mod pilots;
mod context;
#[cfg(test)]
mod tests;

use statics::*;
use context::GpuCtx;
use types::BeingState;
use genome_io::{save_genome, load_genome, population_seed_for};

use universe::gpu::evolution::{Genome, EvoParams, GenResult, evolve, gen_stats, linear_slope};
use universe::world::VoxelWorld;
use universe::evolution::dna_genesis::default_kan_knots;
use universe::engine::SovereignEngine;

// ── Tipos locais (espelhando gpu_loop.rs) ────────────────────────────────────
//
// RetinaParamsGpu/PoseRaw, uniform_entry/storage_entry, pos_of e make_physics_state
// eram cópias LOCAIS idênticas às de gpu_loop.rs e cortex_evolve.rs — deduplicadas
// em E3 (REFACTOR_PLAN.md §2.3) para src/gpu/retina_types.rs, src/gpu/core.rs e
// src/gpu/physics.rs. Os três bins agora leem da mesma fonte. `BeingState` moveu
// para types.rs (usada por main.rs e context.rs); GpuCtx moveu para context.rs.
// Os demais imports antigos (SENSORY_WAVE_SIZE, RETINA_*, Pose, RetinaGpu, BrainGpu,
// SlowBrainGpu, PerceptionGpu, PhysicsGpu/PhysicsStateGpu/WheelGpu, RacingLine, pitch,
// RetinaPerception, Channel/channel_source_for/channel_name_for, KAN_OUT_KNOTS) só
// eram usados dentro de `GpuCtx` — ficaram em context.rs/audit.rs, não aqui.
use universe::gpu::physics::{pos_of, make_physics_state};


// ── Persistência de genomas ───────────────────────────────────────────────────
//
// save_genome/load_genome/population_seed_for moveram para genome_io.rs (E3), com
// cabeçalho versionado v2 (magic "UGEN"+versão) — ver `use genome_io::{..}` no topo.
// VOXEL_WALL, WALL_NEAR_M, todos os statics de flag (FUEL_ON..PILOTO_SENSOR),
// AtomicF32 e life_cap() moveram para statics.rs (E3) — ver `use statics::*;` no
// topo do arquivo, com tabela de "quem lê / quem escreve" no doc do módulo.

// ── Ponto de entrada ──────────────────────────────────────────────────────────

fn main() {
    // Força ICD Vulkan da Radeon (evita SIGSEGV com múltiplos ICDs).
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists()
        && std::env::var("VK_ICD_FILENAMES").is_err()
    {
        unsafe { std::env::set_var("VK_ICD_FILENAMES", icd_radeon); }
    }

    // ── Parsing de args ────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut n_seres:  usize = 60;
    let mut n_ticks:  usize = 8_000;   // 8.000 ticks = 125s @ 64Hz por geração
    let mut n_gens:   usize = 100;
    let mut retomar:  bool  = false;
    let mut out_dir   = ".".to_string();
    // Parâmetros evolutivos ajustáveis via CLI (padrão conservador para fine-tuning).
    // Para exploração inicial/escape de platô: use --sigma 0.15 --p-mutate 0.05
    let mut evo_sigma:    f32  = 0.05;
    let mut evo_p_mutate: f32  = 0.02;
    // Semente da população inicial. Sem isto, repetições da mesma condição são
    // idênticas e não distinguem efeito de sorte — que é o ponto de repetir.
    let mut pop_seed: u64 = 0;
    // Custo de encostar na parede, em metros de fitness por segundo colado.
    //
    // Hoje bater na parede NÃO CUSTA NADA neste caminho: `grep -E "wall|death|crash"` em
    // gpu_evolve.rs devolve zero, e a única morte é por estagnação. A política ótima vira
    // "empurra, trava, dá ré, empurra de novo" — medido no replay: 33% dos quadros em ré e
    // o arco CONGELADO em 336,2 m por centenas de ticks, com governor/sentinel em 0%.
    //
    // Isto NÃO ensina a desviar. Não há regra dizendo para onde virar, nem freio automático.
    // Só torna a parede cara, que é o que faz a competência de evitá-la valer a pena. A
    // inteligência tem de aparecer sozinha; o mundo é que precisa ter consequência.
    let mut wall_cost: f32 = 0.0;
    let mut brain_skip: u32 = 1;
    /// Ticks de vida consumidos A MAIS por tick colado na parede. 0 = sem custo.
    let mut wall_life: i64 = 0;
    /// Ticks de vida consumidos A MAIS por tick PARADO (stagnation_ticks > 128). 0 = sem custo.
    let mut stag_life: i64 = 0;
    /// Todos os seres nascem no mesmo ponto. Torna a fitness comparável entre indivíduos
    /// — sem isso ela mede a largada, não o cérebro.
    let mut same_start = false;
    /// Ticks de vida ganhos por METRO avançado. 0 = vida dada de graça (comportamento antigo).
    ///
    /// Medido: com vida fixa, a população converge para RASTEJAR — 2,2 m/s de média com
    /// 37,7 m/s disponíveis, e mortalidade 0,0%. Pior: com MAIS tempo eles ficam mais
    /// lentos (1,48 m/s em 24k ticks contra 2,96 m/s em 8k). A fitness é distância em
    /// tempo fixo, então devagar-e-seguro ganha de rápido-e-bate: não existe pressa.
    ///
    /// Ganhar vida avançando é a tese do projeto aplicada ao corpo: sobreviver EXIGE
    /// progredir. Não diz ao ser como dirigir nem para onde virar — diz que parar é morrer.
    let mut life_earn: f32 = 0.0;
    /// Largura da pista em metros. Estava fixa em 18,0 no código. A arqueologia de git
    /// mostrou que a versão que o Leonardo lembra rodava com 28 m; pista mais larga deixa
    /// carregar velocidade na curva, que é exatamente onde os seres perdem tempo hoje.
    let mut track_width: f32 = 18.0;
    /// `--fitness dist` volta ao hodômetro antigo. Padrão: progresso na pista.
    /// `checkpoint` (padrão) · `prog` · `dist`. Ver o `match` que monta a fitness.
    let mut modo_fitness = String::from("checkpoint");
    /// Alisamento do volante. 0,5 é o histórico; 1,0 remove a histerese.
    let mut steer_alpha: f32 = 0.5;
    let mut fuel_on = false;
    let mut refuel_laps: u32 = 1;
    let mut fuel_tank: f32 = 1.0;
    let mut setor_paga = true;
    /// `--auditar-visao`: mede se os canais sensoriais respondem à pista real.
    let mut auditar_visao = false;
    let mut portal_curva = false;
    let mut slow_steer_auth: f32 = 0.20;
    /// Teto de vida acumulável em ticks (0 = usar o padrão de 1920).
    let mut life_cap_arg: i64 = 0;
    let mut evo_n_elite:  usize = 3;
    let mut evo_tourney:  usize = 3;
    // gen_offset: número de geração inicial para relatório (útil ao retomar).
    let mut gen_offset:   usize = 0;
    /// `--tau-hetero`: banco de τ log-uniforme em [25 ms, 4 s], um por canal do CfC.
    ///
    /// Sem a flag todos os 96 canais ficam em τ = 2,0 s — o valor único que o cérebro
    /// sempre teve. Isso importa porque os canais filtrados são justamente parede,
    /// curvatura, velocidade e alinhamento: a 15 m/s um filtro de 2 s entrega ao volante
    /// a percepção de 30 m atrás, e andar rápido fica punido pelo próprio cérebro.
    let mut tau_hetero = false;
    /// `--centro-simetrico`: recuar SUBTRAI da fitness de centralização.
    let mut centro_simetrico = false;
    /// `--out-beta`: força da fadiga de saída. Padrão = o histórico.
    let mut out_beta: f32 = universe::gpu::brain::OUT_BETA_PADRAO;
    /// `--piloto <g>`: ganho do piloto fantasma. Diagnóstico, não produção.
    let mut piloto: f32 = 0.0;
    let mut piloto_thr: f32 = 0.0;
    let mut steer_fixo: f32 = 0.0;
    let mut steer_escala: f32 = 0.0;
    let mut piloto_sensor: f32 = 0.0;
    let mut thr_fixo:   f32 = 0.0;
    // `--ativacao`: liga métricas de ativação L1/L2 + guardas de runtime (E4).
    let mut ativacao_on = false;
    // `--ativacao-n <N>`: amostra a cada N ticks (padrão 64).
    let mut ativacao_n: u32 = 64;
    {
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--seres"     => { i += 1; if i < args.len() { n_seres       = args[i].parse().unwrap_or(60); } }
                "--ticks"     => { i += 1; if i < args.len() { n_ticks       = args[i].parse().unwrap_or(8_000); } }
                "--gens"      => { i += 1; if i < args.len() { n_gens        = args[i].parse().unwrap_or(100); } }
                "--sigma"     => { i += 1; if i < args.len() { evo_sigma     = args[i].parse().unwrap_or(0.05); } }
                "--p-mutate"  => { i += 1; if i < args.len() { evo_p_mutate  = args[i].parse().unwrap_or(0.02); } }
                "--sem-lento" => { SLOW_ABLATED.store(true, std::sync::atomic::Ordering::Relaxed); }
                // Banco de τ log-uniforme em [25 ms, 4 s] em vez do τ = 2,0 s uniforme.
                // Sem a flag o comportamento é o histórico — é o braço de controle.
                "--tau-hetero" => { tau_hetero = true; }
                // Cobra o recuo na fitness `centro` (ver a acumulação de `center_prog`).
                "--centro-simetrico" => { centro_simetrico = true; }
                // Força da fadiga de SAÍDA. 0,85 (padrão) faz comando sustentado sair a
                // 15% da intenção em ~1 s; 0,0 desliga. Ver `BrainParams::out_beta`.
                "--out-beta"  => { i += 1; if i < args.len() { out_beta = args[i].parse().unwrap_or(out_beta); } }
                // Piloto fantasma: ganho do controlador geométrico. 0 = cérebro no comando.
                "--piloto"    => { i += 1; if i < args.len() { piloto = args[i].parse().unwrap_or(0.0); } }
                "--piloto-thr" => { i += 1; if i < args.len() { piloto_thr = args[i].parse().unwrap_or(0.45); } }
                "--steer-fixo" => { i += 1; if i < args.len() { steer_fixo = args[i].parse().unwrap_or(0.0); } }
                "--steer-escala" => { i += 1; if i < args.len() { steer_escala = args[i].parse().unwrap_or(0.0); } }
                "--piloto-sensor" => { i += 1; if i < args.len() { piloto_sensor = args[i].parse().unwrap_or(0.0); } }
                "--thr-fixo"   => { i += 1; if i < args.len() { thr_fixo   = args[i].parse().unwrap_or(0.0); } }
                "--seed"      => { i += 1; if i < args.len() { pop_seed      = args[i].parse().unwrap_or(0); } }
                "--wall-cost" => { i += 1; if i < args.len() { wall_cost     = args[i].parse().unwrap_or(0.0); } }
                "--brain-skip"=> { i += 1; if i < args.len() { brain_skip    = args[i].parse().unwrap_or(1).max(1); } }
                "--wall-life" => { i += 1; if i < args.len() { wall_life     = args[i].parse().unwrap_or(0); } }
                "--stag-life-drain" => { i += 1; if i < args.len() { stag_life = args[i].parse().unwrap_or(0); } }
                "--same-start"=> { same_start = true; }
                "--life-earn" => { i += 1; if i < args.len() { life_earn     = args[i].parse().unwrap_or(0.0); } }
                "--width"     => { i += 1; if i < args.len() { track_width   = args[i].parse().unwrap_or(18.0); } }
                "--fitness"   => { i += 1; if i < args.len() { modo_fitness = args[i].clone(); } }
                "--steer-alpha"=>{ i += 1; if i < args.len() { steer_alpha = args[i].parse().unwrap_or(0.5); } }
                "--fuel"      => { fuel_on = true; }
                "--fuel-tank" => { i += 1; if i < args.len() { fuel_tank = args[i].parse().unwrap_or(1.0); fuel_on = true; } }
                "--refuel-lap-only" => { setor_paga = false; }
                "--auditar-visao" => { auditar_visao = true; }
                "--portal-curva" => { portal_curva = true; }
                "--slow-steer" => { i += 1; if i < args.len() { slow_steer_auth = args[i].parse().unwrap_or(0.20); } }
                "--refuel-laps"=>{ i += 1; if i < args.len() { refuel_laps = args[i].parse().unwrap_or(1).max(1); } }
                "--life-cap"  => { i += 1; if i < args.len() { life_cap_arg  = args[i].parse().unwrap_or(0); } }
                "--n-elite"   => { i += 1; if i < args.len() { evo_n_elite   = args[i].parse().unwrap_or(3); } }
                "--tourney"   => { i += 1; if i < args.len() { evo_tourney   = args[i].parse().unwrap_or(3); } }
                "--gen-offset"=> { i += 1; if i < args.len() { gen_offset    = args[i].parse().unwrap_or(0); } }
                "--ativacao"   => { ativacao_on = true; }
                "--ativacao-n" => { i += 1; if i < args.len() { ativacao_n = args[i].parse().unwrap_or(64).max(1); } }
                "--retomar"   => { retomar = true; }
                "--out"       => { i += 1; if i < args.len() { out_dir       = args[i].clone(); } }
                _ => {}
            }
            i += 1;
        }
    }

    let inicio = if retomar { "retomando de checkpoint" } else { "pesos ALEATÓRIOS (σ=0.30, N(0,σ))" };
    println!("=== gpu_evolve: {} seres · {} ticks/gen · {} gens ===", n_seres, n_ticks, n_gens);
    println!("  Inicio: {}", inicio);
    println!("  Evolução: sigma={:.3}  p_mutate={:.3}  n_elite={}  tourney={}", evo_sigma, evo_p_mutate, evo_n_elite, evo_tourney);
    println!("  SEM muletas motoras (escape_ticks, curvature_governor)");
    println!("  Fitness: distância contínua em metros (sem degrau de volta inteira)");
    println!();

    // ── Mundo ─────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let world = VoxelWorld::new(9001, track_width);
    let lap_dist = world.lap_distance as f64;
    println!("mundo: {}×{} · {} wp · volta={:.0}m  [{:.0}ms]",
        world.size, world.size, world.waypoints_world.len(), lap_dist,
        t0.elapsed().as_secs_f64() * 1000.0);
    println!();

    // ── GPU ───────────────────────────────────────────────────────────────────
    let t1 = Instant::now();
    let mut ctx = match GpuCtx::new(&world, n_seres) {
        Some(c) => c,
        None => {
            eprintln!("ERRO: sem GPU utilizável. Abortando.");
            std::process::exit(1);
        }
    };
    ctx.brain_skip = brain_skip;
    // Fitness do melhor da geração anterior — o mesmo genoma reaparece sem mutação no
    // slot 0 da geração seguinte, então a comparação é do genoma consigo mesmo.
    let mut elite_fitness_anterior: Option<f64> = None;
    let mut elite_reprod: Vec<f64> = Vec::new();
    WALL_LIFE_DRAIN.store(wall_life, Ordering::Relaxed);
    STAG_LIFE_DRAIN.store(stag_life, Ordering::Relaxed);
    LIFE_EARN.store(life_earn, Ordering::Relaxed);
    STEER_ALPHA.store(steer_alpha.clamp(0.05, 1.0), Ordering::Relaxed);
    FUEL_ON.store(fuel_on, Ordering::Relaxed);
    REFUEL_LAPS.store(refuel_laps, Ordering::Relaxed);
    FUEL_TANK.store(fuel_tank.max(0.01), Ordering::Relaxed);
    SETOR_PAGA.store(setor_paga, Ordering::Relaxed);
    AUDITAR.store(auditar_visao, Ordering::Relaxed);
    PORTAL_CURVA.store(portal_curva, Ordering::Relaxed);
    SLOW_STEER_AUTH.store(slow_steer_auth, Ordering::Relaxed);
    if life_cap_arg > 0 { LIFE_CAP_TICKS.store(life_cap_arg, Ordering::Relaxed); }
    ctx.brain.out_beta = out_beta;
    PILOTO_GANHO.store(piloto, Ordering::Relaxed);
    PILOTO_THR.store(if piloto_thr > 0.0 { piloto_thr } else { 0.45 }, Ordering::Relaxed);
    STEER_FIXO.store(steer_fixo, Ordering::Relaxed);
    STEER_ESCALA.store(steer_escala, Ordering::Relaxed);
    PILOTO_SENSOR.store(piloto_sensor, Ordering::Relaxed);
    ATIVACAO_ON.store(ativacao_on, Ordering::Relaxed);
    ATIVACAO_N.store(ativacao_n, Ordering::Relaxed);
    if ativacao_on {
        println!("[ativacao] métricas de ativação L1/L2 LIGADAS — amostra a cada {ativacao_n} ticks");
    }
    if piloto_sensor != 0.0 {
        println!("PILOTO SENSORIAL (ganho {piloto_sensor:+.2}) — sem cérebro, só raios de \
                  parede. Mede o TETO DA INFORMAÇÃO, não o modelo.");
    }
    THR_FIXO.store(thr_fixo, Ordering::Relaxed);
    if steer_fixo != 0.0 || thr_fixo != 0.0 {
        println!("OVERRIDE BRUTO: volante={steer_fixo:+.2} acelerador={thr_fixo:.2} \
                  — o cérebro NÃO comanda nada. Isto mede o CORPO.");
    }
    if piloto != 0.0 {
        println!("PILOTO FANTASMA ligado (ganho {piloto:+.1}) — o cérebro NÃO comanda o volante. \
                  Isto mede o CORPO, não o modelo.");
    }
    if (out_beta - universe::gpu::brain::OUT_BETA_PADRAO).abs() > 1e-6 {
        println!("fadiga de saída: out_beta={out_beta:.2} (padrão {:.2}) — comando sustentado \
                  sai a {:.0}% da intenção",
                 universe::gpu::brain::OUT_BETA_PADRAO, 100.0 * (1.0 - out_beta).max(0.0));
    }
    println!("GpuCtx pronto  [{:.0}ms]", t1.elapsed().as_secs_f64() * 1000.0);

    // ── Genomas iniciais ───────────────────────────────────────────────────────
    let mut genomes: Vec<Genome> = if retomar {
        // Procura o arquivo de melhor genoma mais recente.
        let mut latest: Option<(usize, String)> = None;
        for g in 0..n_gens {
            let path = format!("{}/evolve_best_gen{:04}.bin", out_dir, g);
            if std::path::Path::new(&path).exists() {
                latest = Some((g, path));
            }
        }
        if let Some((gen_id, path)) = latest {
            println!("Retomando a partir de gen {:4}: {}", gen_id, path);
            if let Some(best) = load_genome(&path) {
                // Semente a população a partir do melhor genoma + variações.
                let mut pop = Vec::with_capacity(n_seres);
                pop.push(best.clone());  // elite exato
                let evo = EvoParams { p_mutate: 0.05, sigma: 0.15, n_elite: 1, tournament_k: 1 };
                while pop.len() < n_seres {
                    let mut filho = best.clone();
                    // Reusa a lógica de mutação do módulo.
                    let seed = pop.len() as u64 ^ (gen_id as u64 * 0xDEAD_BEEF);
                    let noise_src = Genome::random(seed);
                    // Mistura: 70% pai + 30% aleatório (perturba mais forte para diversidade).
                    for (w, n) in filho.l1.iter_mut().zip(noise_src.l1.iter()) {
                        *w = *w * 0.70 + *n * 0.30;
                    }
                    for (w, n) in filho.l2.iter_mut().zip(noise_src.l2.iter()) {
                        *w = *w * 0.70 + *n * 0.30;
                    }
                    pop.push(filho);
                }
                pop
            } else {
                eprintln!("Falha ao carregar arquivo — usando população aleatória.");
                (0..n_seres).map(|i| Genome::random(population_seed_for(pop_seed, i))).collect()
            }
        } else {
            println!("Nenhum arquivo de checkpoint encontrado — população aleatória.");
            (0..n_seres).map(|i| Genome::random(population_seed_for(pop_seed, i))).collect()
        }
    } else {
        // População aleatória do zero.
        // Seed = índice do ser (determinístico e diverso).
        (0..n_seres).map(|i| Genome::random(population_seed_for(pop_seed, i))).collect()
    };

    // Banco de τ heterogêneo: aplicado DEPOIS de montar a população, em qualquer um dos
    // caminhos acima (aleatória, de checkpoint, mista). Um sorteio por ser — τ é parte do
    // genoma desde o nascimento e daí em diante só muda por mutação e seleção.
    //
    // A semente é derivada da mesma função dos pesos, mas com um deslocamento, senão o
    // banco de τ do ser i seria função da mesma sequência que gerou seus pesos.
    if tau_hetero {
        for (i, g) in genomes.iter_mut().enumerate() {
            g.randomize_tau(population_seed_for(pop_seed, i).wrapping_add(0x7A11_0000));
        }
    }

    // Parâmetros evolutivos: configuráveis via CLI, default conservador.
    let evo_params = EvoParams {
        p_mutate:    evo_p_mutate,
        sigma:       evo_sigma,
        n_elite:     evo_n_elite,
        tournament_k: evo_tourney,
    };

    // ── Histórico de medianas (para inclinação ao final) ──────────────────────
    let mut hist_med:  Vec<f64> = Vec::with_capacity(n_gens);
    let mut hist_max:  Vec<f64> = Vec::with_capacity(n_gens);
    let mut hist_results: Vec<GenResult> = Vec::with_capacity(n_gens);

    // ── KAN knots iniciais: default de saída ──────────────────────────────────
    // Cada genoma carrega seus kan_knots. Usamos os do melhor indivíduo na hora
    // de rodar os ticks (simplificação: todos usam os knots da população como um todo).
    // Futuramente: cada ser usa seus próprios knots.
    let kan_knots_global = default_kan_knots();

    // ── LAÇO PRINCIPAL ────────────────────────────────────────────────────────
    let t_total = Instant::now();
    for generation_idx in 0..n_gens {
        // gen_abs = número absoluto da geração (para relatório e arquivo de save).
        let gen_abs = gen_offset + generation_idx;
        // Limita a memória de AUDIT_ACUM (risco R6 do REFACTOR_PLAN.md): sem isto,
        // uma bateria de 100 gerações × 8.000 ticks acumula ~166 MB sem nunca liberar.
        // Efeito colateral aceito: o relatório final passa a refletir só a ÚLTIMA
        // geração — que é o dado útil (população evoluída, não diluída pelos
        // indivíduos aleatórios do início).
        audit::nova_geracao();
        // ── REPLAY do líder ────────────────────────────────────────────────────
        // O `gpu_evolve` não gravava replay nenhum: `grep ReplayWriter` devolvia zero.
        // Sem isso não há como o Leonardo VER o que os seres fazem — só médias. Grava a
        // 16 Hz (1 quadro a cada 4 ticks) o ser que está na frente.
        let replay_path = std::path::PathBuf::from(&out_dir).join(format!("gen{gen_num:04}.replay",
            gen_num = gen_abs));
        let _ = std::fs::create_dir_all(&out_dir);
        // TODOS os seres, não só o líder.
        //
        // Gravar só quem está na frente esconde exatamente o que interessa: por que os
        // OUTROS 47 pararam, onde cada um travou, se falham no mesmo ponto ou em pontos
        // diferentes. Com um ser por replay a resposta para "por que só 2 de 60 passam" é
        // inalcançável por construção.
        let n_grav = n_seres.min(255);
        let mut replay_writer = universe::replay::ReplayWriter::create(
            &replay_path, universe::replay::SOURCE_CORTEX, n_grav as u8, 64.0, 9001u64,
        ).ok();
        let t_gen = Instant::now();

        // ── Aplica genomas nos engines ─────────────────────────────────────────
        let mut engines: Vec<SovereignEngine> = (0..n_seres).map(|_| SovereignEngine::new()).collect();
        for (i, (eng, genome)) in engines.iter_mut().zip(genomes.iter()).enumerate() {
            genome.apply_to_engine(eng);
        }
        ctx.upload_engines(&engines);
        // Banco de τ: vai separado porque τ vive no genoma, não no `SovereignEngine`.
        // Uma vez por geração, como os pesos — τ é herdado e fixo na vida do ser.
        ctx.brain.upload_tau(&Genome::pack_taus(&genomes), n_seres);

        // ── Reseta CfC entre gerações ──────────────────────────────────────────
        // reset_cfc() reseta só percept_cfc (percepção visual).
        // CORREÇÃO (2026-08-11): buf_cfc (estado recorrente, 97 f32 por ser) e
        // buf_fatigue (fadiga adaptativa, 192 f32 por ser) residem na GPU e nunca eram
        // zerados entre gerações — o filho herdava o estado CfC do pai em silêncio.
        // Isso torna a reavaliação do campeão não-determinística: o elite na gen N+1
        // começa com o estado CfC acumulado da gen N, não com zeros.
        ctx.reset_cfc(n_seres);
        for i in 0..n_seres {
            ctx.brain.reset_cfc(i);
        }

        // ── Estado inicial dos seres ────────────────────────────────────────────
        // Distribui ao longo dos waypoints (diversidade de posição inicial).
        let n_wp = world.waypoints_world.len();
        let mut states: Vec<BeingState> = (0..n_seres).map(|i| {
            // Onde cada ser nasce.
            //
            // Espalhar a população pelos waypoints parecia razoável — mais trechos vistos,
            // menos colisão na largada. Só que a fitness é distância percorrida, e trechos
            // de pista NÃO são igualmente difíceis: quem nasce numa reta longa acumula
            // muito, quem nasce antes de uma curva fechada trava.
            //
            // Consequência medida: o MESMO genoma, reavaliado na geração seguinte (o elite
            // passa sem mutação), rendia em média 72% do anterior — e quanto MAIOR o feito,
            // mais perdia (536 m → 287 m, 54%). Isso é regressão à média, a assinatura de
            // seleção sobre RUÍDO: o campeão não sabia dar volta, tinha nascido no lugar
            // certo. Como na geração seguinte ele nasce noutro ponto, não há nada a herdar,
            // e por isso o número de completadores nunca cresce.
            let wi = if same_start { 0 } else { (i * n_wp / n_seres).min(n_wp - 1) };
            let wp = world.waypoints_world[wi];
            let wp_next = world.waypoints_world[(wi + 1).min(n_wp - 1)];
            let dx = wp_next.x - wp.x;
            let dz = wp_next.y - wp.y;
            let len = (dx*dx + dz*dz).sqrt().max(1e-6);
            let fwd = [dx / len, 0.0, dz / len];
            let pos = [wp.x, 0.0, wp.y];
            BeingState {
                phys: make_physics_state(pos, fwd),
                prev_steer: 0.0,
                stagnation_ticks: 0,
                wall_ticks: 0,
                center_prog: 0.0,
                sensor_steer: 0.0,
                sensor_thr: 0.0,
                sensor_brk: 0.0,
                wp_near: 0,
                fuel: FUEL_TANK.load(Ordering::Relaxed) as f64,
                ticks_alive: 0,
                // Com vida ganha, o ser nasce com pouco fôlego (10 s) e precisa
                // conquistar o resto. Sem isso, ele teria a geração inteira garantida e o
                // mecanismo não teria efeito nenhum nos primeiros minutos.
                life_left: if LIFE_EARN.load(Ordering::Relaxed) > 0.0 { LIFE_START as f64 }
                           else { n_ticks as f64 },
                reverse_timer: 0.0,
                post_reverse_lockout: 0.0,
                joy_tick: 0.0,
                joy_streak: 0.0,
                edge_row6: 0.0,
                edge_row7: 0.0,
                phantom_d_min: 30.0,
                dist_wall_l:  40.0,
                dist_wall_r:  40.0,
                dist_wall_f:  50.0,
            }
        }).collect();

        // ── Odômetro ───────────────────────────────────────────────────────────
        let mut dist_acum: Vec<f64> = vec![0.0; n_seres];
        // Progresso AO LONGO DA PISTA, monotônico e com tratamento de cruzamento de linha.
        //
        // A fitness era o hodômetro puro (`dist_acum`), que conta qualquer deslocamento:
        // fora da pista, em círculo, indo e voltando. Seguir o circuito nunca deu vantagem
        // nenhuma — e o replay mostrou o campeão indo RETO e saindo da pista na primeira
        // curva. Ele não errava a curva; fazer a curva não valia nada.
        let mut prog_acum: Vec<f64> = vec![0.0; n_seres];
        // Checkpoint alcançado por cada ser, EM ORDEM. É o que dá motivo para terminar
        // a volta.
        //
        // Antes a fitness premiava "andar e ficar vivo": um ser parado aos 265 m tinha a
        // mesma nota de um que tentasse ir além, então não havia gradiente que levasse ao
        // fim do circuito. E rodar em círculo dentro da pista não custava nada.
        //
        // O checkpoint só conta se for o PRÓXIMO da sequência. Voltar não desconta, mas
        // também não adianta: repetir um trecho não rende checkpoint nenhum. Assim o único
        // caminho para subir na fitness é AVANÇAR no traçado.
        let mut checkpoint: Vec<u32> = vec![0; n_seres];
        // Portais de recompensa NAS CURVAS, não em intervalos iguais.
        //
        // Leonardo: *"cada curva dá combustível, logo eles vão aprender fazer curva"*.
        //
        // Setor uniforme paga por andar; portal na curva paga por CURVAR. Como a coisa
        // que eles não sabem fazer é exatamente sair da curva, o pagamento vai onde a
        // competência falta. E os trechos retos deixam de render nada, então acelerar em
        // linha reta para de ser estratégia.
        //
        // Um portal por vértice com curva acima de 8°, mais a linha de chegada.
        let portais: Vec<f32> = {
            let wps = &world.waypoints_world;
            let mut acc = 0.0f32;
            let mut v: Vec<f32> = vec![0.0];
            for i in 1..wps.len().saturating_sub(1) {
                acc += wps[i].distance(wps[i - 1]);
                let a = (wps[i] - wps[i - 1]).normalize_or_zero();
                let b = (wps[i + 1] - wps[i]).normalize_or_zero();
                let ang = a.dot(b).clamp(-1.0, 1.0).acos().to_degrees();
                if ang > 8.0 { v.push(acc); }
            }
            v
        };
        if generation_idx == 0 {
            println!("  [portais] {} portais em curvas > 8° (uniforme seriam {})",
                     portais.len(), N_CHECKPOINTS);
        }
        let mut voltas_reais: Vec<u32> = vec![0; n_seres];
        let mut prog_prev: Vec<f32> = (0..n_seres)
            .map(|i| world.get_track_progress(states[i].phys.pos[0], states[i].phys.pos[2]))
            .collect();
        let mut pos_prev: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

        // Tick de aquecimento (compilação JIT — descartado).
        let _ = ctx.tick(&mut states, n_seres, 1.0/64.0, 0, &kan_knots_global);

        // ── N ticks de simulação ───────────────────────────────────────────────
        let dt = 1.0_f32 / 64.0;
        let mut abortado = false;
        for tick_i in 0..n_ticks {
            let ok = ctx.tick(&mut states, n_seres, dt, tick_i + 1, &kan_knots_global);
            if !ok {
                eprintln!("[gen {:4}] ERRO no tick {} — abortando geração.", gen_abs, tick_i);
                abortado = true;
                break;
            }
            // Odômetro: distância euclidiana entre pos[t-1] e pos[t].
            // Com dt=1/64s e v≤60m/s, passo ≤0.94m → erro ≈ arclength < 0.1%.
            // Quadro de replay do líder, a 16 Hz.
            if tick_i % 4 == 0 {
                if let Some(ref mut rw) = replay_writer {
                  for lider in 0..n_grav {
                    let ph = &states[lider].phys;
                    let vel = (ph.vel[0].powi(2) + ph.vel[2].powi(2)).sqrt();
                    let rf = universe::replay::ReplayFrame {
                        pos: ph.pos,
                        fwd: [ph.fwd[0], ph.fwd[2]],
                        speed: vel,
                        yaw_rate: ph.av_y,
                        arc_prog: dist_acum[lider] as f32,
                        laps: 0,
                        gear: ph.current_gear as u8,
                        // Vida restante no lugar do HP: é o recurso que este caminho
                        // realmente consome, e é o que explica o ser parar.
                        hp_norm: ((states[lider].life_left.max(0.0) as f32
                                   / life_cap() as f32).min(1.0) * 127.0) as u8,
                        fuel_norm: 127,
                        tick: tick_i as u32,
                        // COMANDOS: sem eles não dá para separar "o cérebro não manda
                        // virar" de "o carro não consegue virar" — e é exatamente essa a
                        // pergunta agora que sabemos que todos param na primeira curva.
                        // `intent_*` é a decisão neural crua; `final_*` é o que a física
                        // recebeu depois do KAN, da modulação lenta e do trecho de ré.
                        final_thr: ph.throttle,
                        final_str: ph.steer,
                        final_brk: ph.brake_cmd,
                        // Distâncias às paredes: agora gravadas de verdade (Conserto 4, 2026-08-12).
                        // Antes chegavam zeradas via ..Default::default(). Com estes valores é
                        // possível saber exatamente o que o cérebro via ao parar na curva.
                        dist_wall_l: states[lider].dist_wall_l,
                        dist_wall_r: states[lider].dist_wall_r,
                        dist_wall_f: states[lider].dist_wall_f,
                        // `intent_*` fica zerado aqui de propósito: a intenção neural
                        // crua vive dentro de `tick()` e não chega a este laço. O que a
                        // física RECEBEU já responde a pergunta atual.
                        ..Default::default()
                    };
                    let _ = rw.write_frame(lider as u8, &rf);
                  }
                }
            }

            for i in 0..n_seres {
                let p1 = pos_of(&states[i].phys);
                let dx = (p1[0] - pos_prev[i][0]) as f64;
                let dz = (p1[2] - pos_prev[i][2]) as f64;
                if states[i].life_left > 0.0 {
                    let passo = (dx*dx + dz*dz).sqrt();
                    dist_acum[i] += passo;

                    // Avanço na pista desde o tick anterior, com o salto da linha de
                    // chegada tratado nos dois sentidos. O clamp a 10% da volta descarta
                    // teleporte numérico sem descartar avanço legítimo.
                    let cur = world.get_track_progress(p1[0], p1[2]);
                    let ld = lap_dist as f32;
                    let mut d = cur - prog_prev[i];
                    if d < -ld * 0.5 { d += ld; } else if d > ld * 0.5 { d -= ld; }
                    let avanco = d.clamp(-ld * 0.1, ld * 0.1) as f64;
                    prog_acum[i] += avanco;
                    prog_prev[i] = cur;

                    // DOPAMINA — canais 126 (joy_tick) e 127 (joy_streak).
                    //
                    // Replicando motor.rs:1666-1677: joy_tick = progress_speed *
                    // surface_factor, joy_streak acumula em 5 s de flow e decai 2× rápido.
                    // progress_speed = speed alinhada / 30 m/s; surface_factor depende do
                    // voxel (estrada=1, grama=0.2, parede=-0.5). O ser deve SENTIR quando
                    // está indo bem para que o cérebro lento (System 2) possa modular.
                    {
                        let p   = &states[i].phys;
                        let spd = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                        let vel_dot = p.vel[0]*p.fwd[0] + p.vel[2]*p.fwd[2];
                        let align = if spd > 0.1 { (vel_dot / spd).clamp(-1.0, 1.0) } else { 1.0 };
                        let progress_speed = (spd * align.max(0.0) / 30.0).clamp(0.0, 1.0);
                        // Voxel sob o ser usando a grade CPU (mesma grade usada para
                        // propriocepção acima — não há custo adicional).
                        let gs  = ctx.grid_size_cpu as usize;
                        let cs  = ctx.cell_size_cpu;
                        let xi  = (p.pos[0] / cs).round() as i32;
                        let zi  = (p.pos[2] / cs).round() as i32;
                        let vox = if xi >= 0 && xi < gs as i32 && zi >= 0 && zi < gs as i32 {
                            ctx.grid_cpu[(zi as usize) * gs + xi as usize]
                        } else { 2u32 };
                        // 0=grama, 1=estrada, 2=parede, 3=buraco (VoxelType ordem em world.rs)
                        let surface_factor: f32 = match vox {
                            1 => 1.0,   // Road
                            0 => 0.2,   // Grass
                            2 => -0.5,  // Wall
                            _ => -1.0,  // Hole (3) e demais
                        };
                        let joy_tick = progress_speed * surface_factor;
                        // Acumula streak: sobe quando fluindo, decai 2× mais rápido quando não.
                        let dt = 1.0_f32 / 64.0;
                        if joy_tick > 0.5 {
                            states[i].joy_streak = (states[i].joy_streak + dt).min(5.0);
                        } else {
                            states[i].joy_streak = (states[i].joy_streak - dt * 2.0).max(0.0);
                        }
                        states[i].joy_tick = joy_tick;
                    }

                    // CENTRALIZAÇÃO: distância ao eixo pelo waypoint mais próximo,
                    // com busca LOCAL (±4) a partir do índice cacheado — o progresso é
                    // monotônico, então o waypoint mais próximo também avança.
                    let wps = &world.waypoints_world;
                    let nwp = wps.len();
                    let mut melhor = states[i].wp_near;
                    let mut best_d = f32::MAX;
                    for off in 0..9 {
                        let idx = (states[i].wp_near + nwp + off - 4) % nwp;
                        let dxx = wps[idx].x - p1[0];
                        let dzz = wps[idx].y - p1[2];
                        let dd = dxx*dxx + dzz*dzz;
                        if dd < best_d { best_d = dd; melhor = idx; }
                    }
                    states[i].wp_near = melhor;
                    let dist_eixo = best_d.sqrt();
                    // Proximidade em [0,1]: 1 no eixo, 0 na meia-largura ou além.
                    let half = (track_width * 0.5).max(1.0);
                    let prox = (1.0 - dist_eixo / half).clamp(0.0, 1.0) as f64;
                    // CATRACA vs SIMÉTRICO.
                    //
                    // A forma original só somava avanço positivo. Recuar não custava nada,
                    // então oscilar no lugar rendia fitness: cada meio-ciclo para frente
                    // somava `avanco*prox` (e `prox≈1` em cima do eixo), cada meio-ciclo
                    // para trás era de graça. "Parar no centro tremendo" vira política
                    // vencedora sem sair do lugar — mesma família do hodômetro que contava
                    // rodar em círculo como volta.
                    //
                    // Simétrico cobra a volta: o saldo passa a ser deslocamento líquido
                    // ponderado. `--centro-simetrico` liga; sem a flag fica a catraca, para
                    // que a comparação seja pareada contra o comportamento medido até aqui.
                    if centro_simetrico {
                        states[i].center_prog += avanco * prox;
                    } else if avanco > 0.0 {
                        states[i].center_prog += avanco * prox;
                    }

                    // Setor atual no circuito, 0..N_CHECKPOINTS-1.
                    // Índice do portal atual: quantos portais já ficaram para trás.
                    let n_port = portais.len() as u32;
                    let setor = if PORTAL_CURVA.load(Ordering::Relaxed) {
                        portais.iter().filter(|&&p| cur >= p).count().saturating_sub(1) as u32
                    } else {
                        ((cur / ld) * N_CHECKPOINTS as f32) as u32
                    };
                    let total = if PORTAL_CURVA.load(Ordering::Relaxed) { n_port } else { N_CHECKPOINTS };
                    let esperado = checkpoint[i] % total.max(1);
                    if setor == esperado {
                        checkpoint[i] += 1;
                        // REABASTECE A CADA SETOR, não só ao fechar volta.
                        //
                        // Reabastecer por volta é recompensa esparsa com ZERO acertos: só
                        // premia quem já sabe dar volta, e ninguém sabe. Medido: tanques
                        // de 0,15 a 2,0 deram resultado idêntico (246 m, 2/24 setores),
                        // porque o combustível matava DEPOIS que o ser já tinha parado —
                        // e matar quem já parou não é pressão seletiva.
                        //
                        // Na `f1-world` isso funcionava porque os seres já navegavam desde
                        // a geração 0 (26 axiomas de direção, removidos no commit 59d0594);
                        // lá o combustível cobrava VELOCIDADE, nunca ensinou a dirigir.
                        //
                        // Um setor a cada ~122 m está sempre ao alcance de quem avança:
                        // a pista passa a PAGAR o avanço, que é o que dá gradiente onde
                        // antes havia só um degrau inalcançável.
                        if FUEL_ON.load(Ordering::Relaxed) && SETOR_PAGA.load(Ordering::Relaxed) {
                            states[i].fuel = FUEL_TANK.load(Ordering::Relaxed) as f64;
                        }
                        if checkpoint[i] % total.max(1) == 0 {
                            voltas_reais[i] += 1;
                            // FECHAR VOLTA REABASTECE — o incentivo que faltava.
                            // Na f1-world era a cada 3 voltas; aqui a cada volta, porque
                            // partindo de ZERO voltas exigir três seria um degrau que a
                            // evolução nunca sobe. O intervalo é `--refuel-laps`.
                            if voltas_reais[i] % REFUEL_LAPS.load(Ordering::Relaxed) == 0 {
                                states[i].fuel = FUEL_TANK.load(Ordering::Relaxed) as f64;
                            }
                        }
                    }
                    let ganho = LIFE_EARN.load(Ordering::Relaxed);
                    if ganho > 0.0 {
                        // Teto: ninguém acumula fôlego infinito numa reta. Sem o teto, um
                        // trecho rápido compraria vida para atravessar a pista inteira
                        // parado depois — o oposto da pressão que se quer criar.
                        states[i].life_left = (states[i].life_left
                            + passo * ganho as f64).min(life_cap() as f64);
                    }
                }
                pos_prev[i] = p1;
            }
        }

        // ── Custo de parede ────────────────────────────────────────────────────
        // Subtrai da fitness o tempo colado na parede, convertido em metros. Com
        // `--wall-cost 0` nada muda e a linha vira só telemetria: é a condição de controle
        // do experimento pareado, no MESMO binário, sem recompilar entre condições.
        let wall_frac: f64 = {
            let total: u64 = states.iter().take(n_seres).map(|s| s.wall_ticks as u64).sum();
            total as f64 / (n_seres as f64 * n_ticks as f64)
        };
        if wall_cost > 0.0 {
            for i in 0..n_seres {
                let segundos_colado = states[i].wall_ticks as f64 / 64.0;
                dist_acum[i] = (dist_acum[i] - segundos_colado * wall_cost as f64).max(0.0);
            }
        }

        // ── Estatísticas da geração ────────────────────────────────────────────
        // A fitness que a evolução enxerga. Progresso por padrão; `--fitness dist`
        // reproduz o hodômetro antigo, para comparar em vez de assumir.
        let fitness: Vec<f64> = match modo_fitness.as_str() {
            // Hodômetro: o comportamento antigo, mantido como controle.
            "dist" => dist_acum.clone(),
            // Progresso puro: avanço no traçado, sem exigir ordem.
            "prog" => prog_acum.iter().map(|p| p.max(0.0)).collect(),
            // Checkpoints (padrão): setores cruzados EM ORDEM, convertidos em metros para
            // ficar na mesma escala das outras, mais o avanço parcial dentro do setor.
            // Girar em círculo não rende setor novo, então não rende fitness.
            // Checkpoints (padrão): setor cruzado EM ORDEM como parte inteira DOMINANTE,
            // progresso contínuo dentro do setor como desempate FINO.
            //
            // A versão anterior somava `prog_acum × 0.01` — 3,7 m contra degraus de 122 m,
            // 33× menor. O desempate sumia no ruído e 48 seres davam só 5 notas distintas:
            // seleção cega, torneio virava sorteio. Medido nesta sessão.
            //
            // Agora cada checkpoint vale LAP inteiro (garante que quem cruzou mais setores
            // em ordem SEMPRE vence), e o progresso monotônico — que já é contínuo e
            // distinto por ser — ordena dentro do setor. Girar em círculo não rende setor
            // nem progresso monotônico, então continua valendo zero.
            // Centralização (--fitness centro): setor cruzado EM ORDEM (domina) +
            // avanço PONDERADO por proximidade do eixo. Cravar deriva → centro baixo →
            // perde para quem modula e segue o traçado. Não recompensa parar
            // (avanço zero → center_prog zero).
            "centro" => (0..n_seres).map(|i| {
                let setor = checkpoint[i] as f64 * lap_dist;
                setor + states[i].center_prog.max(0.0)
            }).collect(),
            _ => (0..n_seres).map(|i| {
                let setor = checkpoint[i] as f64 * lap_dist;
                setor + prog_acum[i].max(0.0)
            }).collect(),
        };

        // ── O campeão se REPRODUZ? ─────────────────────────────────────────────
        //
        // `evolve()` passa os `n_elite` melhores SEM MUTAÇÃO. Logo `genomes[0]` desta
        // geração é literalmente o mesmo genoma que foi o melhor da anterior. Comparar a
        // fitness dele agora com a de então mede uma coisa só, e sem custo nenhum:
        // quanto do "campeão" era genoma e quanto era SORTE.
        //
        // CORREÇÃO (2026-08-11): antes usava `dist_acum[0]` (hodômetro, ~250 m) para
        // comparar com `fitness.max` da geração anterior (checkpoints × lap_dist, ~8800 m).
        // Grandezas diferentes → razão 3%, que não significa nada. Agora ambos os lados
        // são calculados com a mesma fórmula de fitness (seja checkpoints, prog ou dist).
        // O slot 0 de `fitness` corresponde a `genomes[0]` = elite da geração anterior.
        if let Some(anterior) = elite_fitness_anterior {
            let agora = fitness[0];
            let razao = if anterior > 1.0 { agora / anterior } else { f64::NAN };
            println!(
                "  [elite] mesmo genoma: {:.0} → {:.0} (fitness, {:.0}% do anterior) · hodômetro={:.0}m",
                anterior, agora, 100.0 * razao, dist_acum[0]
            );
            elite_reprod.push(razao);
        }

        // DISPERSÃO DA FITNESS — se for zero, a seleção é cega.
        //
        // `med=369m max=369m` no log significa mediana igual ao máximo: todos os 48 seres
        // com a MESMA nota. Torneio entre iguais é sorteio, elitismo entre iguais é
        // arbitrário, e a evolução vira passeio aleatório por mais gerações que se rode.
        // Nenhum incentivo, percepção ou física conserta isso — não há o que ordenar.
        {
            let mut f = fitness.clone();
            f.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (lo, hi) = (f[0], f[f.len()-1]);
            let m = f.iter().sum::<f64>() / f.len() as f64;
            let sd = (f.iter().map(|v| (v-m).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
            let distintos = { let mut u = f.clone();
                u.dedup_by(|a, b| (*a - *b).abs() < 0.5); u.len() };
            println!("  [seleção] fitness {:.0}..{:.0} · desvio {:.2} · valores distintos {}/{}",
                     lo, hi, sd, distintos, f.len());
            if distintos <= 2 {
                println!("            ⇒ SELEÇÃO CEGA: quase todos com a mesma nota.");
            }
        }

        // gen_stats conta "voltas" como fitness/lap_dist. Com a fitness de checkpoint
        // valendo lap_dist por setor, isso vira número inflado (3 setores = "3 voltas").
        // As VOLTAS mostradas têm de vir do progresso real na pista, não da fitness.
        let r = gen_stats(gen_abs, &prog_acum.iter().map(|p| p.max(0.0)).collect::<Vec<_>>(), lap_dist);
        let t_gen_s = t_gen.elapsed().as_secs_f64();

        hist_med.push(r.dist_med);
        hist_max.push(r.dist_max);

        // ── Salva melhor genoma ────────────────────────────────────────────────
        // Usa gen_abs no nome do arquivo — continua sequência do checkpoint.
        // Melhor pela MESMA grandeza que a seleção usa — senão o genoma salvo e o
        // relatório de reprodutibilidade falam de um campeão que a evolução não escolheu.
        let best_idx = fitness.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let save_path = format!("{}/evolve_best_gen{:04}.bin", out_dir, gen_abs);
        if let Err(e) = save_genome(&genomes[best_idx], &save_path) {
            eprintln!("[gen {:4}] AVISO: falha ao salvar genoma: {}", gen_abs, e);
        }

        // ── Relatório por geração ──────────────────────────────────────────────
        // Voltas REAIS: setores cruzados em ordem fechando o circuito. O contador antigo
        // (`≥1V` = hodômetro ≥ 2928 m) chamava de volta qualquer ser que acumulasse a
        // distância, inclusive girando em círculo na mesma curva.
        let v1 = voltas_reais.iter().filter(|&&v| v >= 1).count();
        let v2 = voltas_reais.iter().filter(|&&v| v >= 2).count();
        let cp_med = {
            let mut c: Vec<u32> = checkpoint.clone();
            c.sort_unstable();
            c[c.len() / 2]
        };
        let cp_max = checkpoint.iter().copied().max().unwrap_or(0);
        // O denominador MUDA com `--portal-curva` (43 portais em curvas contra 24 setores
        // uniformes). Imprimir `N_CHECKPOINTS` fixo nos dois casos faz "4/24" parecer melhor
        // que "3/24" quando na verdade são 4/43 (9,3% da volta) contra 3/24 (12,5%) — a
        // condição pior aparecendo como melhor. Por isso a FRAÇÃO vai junto: ela é a única
        // grandeza comparável entre os dois modos.
        let n_portais = if PORTAL_CURVA.load(Ordering::Relaxed) { portais.len() as u32 }
                        else { N_CHECKPOINTS };
        println!("  [setores] mediana {}/{} ({:.1}% da volta) · melhor {}/{} ({:.1}%) · voltas reais: {} seres ≥1, {} ≥2",
                 cp_med, n_portais, 100.0 * cp_med as f32 / n_portais.max(1) as f32,
                 cp_max, n_portais, 100.0 * cp_max as f32 / n_portais.max(1) as f32, v1, v2);

        println!(
            "gen {:4} | med={:7.0}m ({:.2}V) max={:7.0}m ({:.2}V) | ≥1V={} ≥2V={} morto={:.0}% | {:.1}s",
            gen_abs,
            r.dist_med,  r.dist_med  / lap_dist,
            r.dist_max,  r.dist_max  / lap_dist,
            r.n_ge1v, r.n_ge2v,
            r.mortality * 100.0,
            t_gen_s,
        );

        let sem_vida = states.iter().take(n_seres).filter(|s| s.life_left <= 0.0).count();
        if FUEL_ON.load(Ordering::Relaxed) {
            let secos = states.iter().take(n_seres).filter(|s| s.fuel <= 0.0).count();
            let mut f: Vec<f64> = states.iter().take(n_seres).map(|s| s.fuel.max(0.0)).collect();
            f.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("  [combustível] secos {}/{} · tanque restante mediano {:.2} · tanque inicial {:.2}",
                     secos, n_seres, f[f.len()/2], FUEL_TANK.load(Ordering::Relaxed));
        }
        // Velocidade média da população. É a grandeza que a vida-ganha ataca: a distância
        // pode subir só por ter mais tempo, mas a velocidade não mente.
        {
            let mut v: Vec<f64> = (0..n_seres).map(|i| {
                let seg = (states[i].ticks_alive.max(1) as f64) / 64.0;
                dist_acum[i] / seg
            }).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let vmed = v[v.len() / 2];
            let vida_med = states.iter().take(n_seres)
                .map(|s| s.ticks_alive as f64).sum::<f64>() / n_seres as f64;
            {
            // Hodômetro contra progresso: a lacuna entre os dois É o quanto o ser anda sem
            // avançar na pista. Com fitness de hodômetro essa lacuna era invisível e ele
            // podia ir reto para fora do circuito sem perder nada.
            let mut od: Vec<f64> = dist_acum.clone(); od.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let mut pr: Vec<f64> = prog_acum.clone(); pr.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let (odm, prm) = (od[od.len()/2], pr[pr.len()/2]);
            println!("  [pista] hodômetro {:.0} m · progresso {:.0} m · aproveitamento {:.0}%",
                     odm, prm, if odm > 1.0 { 100.0 * prm / odm } else { 0.0 });
        }
        println!("  [veloc] mediana {:.2} m/s ({:.1} km/h) | vida média {:.0}% da geração",
                     vmed, vmed * 3.6, 100.0 * vida_med / n_ticks as f64);
        }
        println!("  [parede] {:.1}% dos ticks colado (<{:.0} m) | sem vida: {}/{} | custo={:.0} m/s",
                 100.0 * wall_frac, WALL_NEAR_M, sem_vida, n_seres, wall_cost);


        // ── O lento merece existir? ────────────────────────────────────────────
        // Razão de participação das três saídas: perto de 3 = três decisões distintas;
        // perto de 1 = um canal usando três nomes, e crescer o órgão seria só espaço.
        if !ctx.slow_samples.is_empty() {
            let samples = std::mem::take(&mut ctx.slow_samples);
            let pr = universe::math::linalg::participation_ratio(&samples);
            let amp: Vec<f32> = (0..3).map(|k| {
                let mx = samples.iter().map(|r| r[k]).fold(f32::MIN, f32::max);
                let mn = samples.iter().map(|r| r[k]).fold(f32::MAX, f32::min);
                mx - mn
            }).collect();
            match pr {
                Some(v) => println!(
                    "  [lento] PR={:.2}/3 ({:.0}% da capacidade) | amplitude {:.3} {:.3} {:.3} | n={}",
                    v, 100.0 * v / 3.0, amp[0], amp[1], amp[2], samples.len()),
                None => println!("  [lento] samples insuficientes (n={})", samples.len()),
            }
        }

        // ── Ativação L1/L2 + guardas de runtime (E4, `--ativacao`) ─────────────
        //
        // `std::mem::take` limpa `ctx.l1_samples`/`l2_samples`/`policy_samples`/
        // `nan_seres` no ato de consumir — mesmo padrão de `slow_samples` acima:
        // sem isso as amostras se acumulariam sem limite numa bateria longa
        // (risco R6 do REFACTOR_PLAN.md), e o relatório passaria a misturar
        // gerações diferentes em vez de mostrar só a que acabou de rodar.
        if ATIVACAO_ON.load(Ordering::Relaxed) {
            let l1 = std::mem::take(&mut ctx.l1_samples);
            let l2 = std::mem::take(&mut ctx.l2_samples);
            let policy = std::mem::take(&mut ctx.policy_samples);
            let nan_seres = std::mem::take(&mut ctx.nan_seres);

            // Pesos de L1 do CAMPEÃO desta geração (`best_idx`, calculado acima a
            // partir da fitness) — mesma fonte que o bloco `[tau]` logo abaixo usa
            // para `campeao.tau`. Só CPU: `Genome::l1` já está em memória, sem
            // readback de GPU nenhum.
            let l1_w = &genomes[best_idx].l1;

            use universe::analysis::activation::collect_activation_stats;
            use universe::config::iara::brain::{L1_NEURONS, L2_NEURONS};

            let (l1_stats, l1_alarms) = collect_activation_stats(
                "L1", &l1, Some(l1_w), L1_NEURONS, None);
            let (l2_stats, l2_alarms) = collect_activation_stats(
                "L2", &l2, None, L2_NEURONS, Some(&policy));

            for (stats, alarms) in [(&l1_stats, &l1_alarms), (&l2_stats, &l2_alarms)] {
                universe::analysis::activation::print_stats_report(stats, alarms);
                let n_neurons = stats.neurons.len().max(1);
                let dead_pct: f32 = stats.neurons.iter().map(|s| s.dead_pct).sum::<f32>()
                    / n_neurons as f32;
                let saturad_pct: f32 = stats.neurons.iter().map(|s| s.saturated_pct).sum::<f32>()
                    / n_neurons as f32;
                println!(
                    "  [ativacao:{}] amostras={} dead_pct={:.1}% saturad_pct={:.1}% \
                     raio_espectral={:.3} rank_efetivo={}/{} alarmes={}",
                    stats.layer_name, stats.n_samples, dead_pct, saturad_pct,
                    stats.spectral_radius, stats.effective_rank, n_neurons, alarms.len(),
                );
            }
            if let Some(pe) = l2_stats.policy_entropy {
                println!(
                    "  [ativacao:L2] entropia_politica={:.3} nats (limiar {:.2}) · ser 0 = campeão herdado",
                    pe, universe::math::thresholds::POLICY_ENTROPY_MIN,
                );
            }
            if !nan_seres.is_empty() {
                println!(
                    "  [ativacao] ALARME: NaN/Inf detectado no estado L1 de {} ser(es) nesta geração",
                    nan_seres.len(),
                );
            }
        }

        // ── Banco de τ ────────────────────────────────────────────────────────
        //
        // A pergunta que esta linha responde não é "τ está ligado?", é "a SELEÇÃO quer
        // τ variado?". Se a dispersão do campeão encolher geração após geração, a
        // população está convergindo para um τ só e a heterogeneidade não paga aqui —
        // resultado negativo legítimo, e barato de ver. Se crescer, paga.
        //
        // Reporta em SEGUNDOS (τ = 1/d) porque é a grandeza com significado físico:
        // dá para comparar direto com o tempo que o ser leva para atravessar a curva.
        {
            let campeao = &genomes[best_idx];
            let mut taus: Vec<f32> = campeao.tau.iter().map(|&d| 1.0 / d).collect();
            taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p10 = taus[taus.len() / 10];
            let p50 = taus[taus.len() / 2];
            let p90 = taus[taus.len() * 9 / 10];
            // Dispersão da população inteira, não só do campeão: um elite com τ espalhado
            // dentro de uma população uniforme seria sorte, não seleção.
            let disp_pop: f32 = genomes.iter().map(|g| {
                let mut t: Vec<f32> = g.tau.iter().map(|&d| 1.0 / d).collect();
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                t[t.len() * 9 / 10] / t[t.len() / 10].max(1e-6)
            }).sum::<f32>() / genomes.len() as f32;
            println!(
                "  [tau] campeão τ p10 {:.3}s · mediana {:.3}s · p90 {:.3}s (espalha {:.1}×) | população {:.1}×",
                p10, p50, p90, p90 / p10.max(1e-6), disp_pop);
        }

        // O genoma que vai para o slot 0 da próxima geração é o melhor DESTA.
        elite_fitness_anterior = Some(
            fitness.iter().cloned().fold(f64::MIN, f64::max)
        );

        hist_results.push(r);

        // ── Evolução (seleção + mutação) ───────────────────────────────────────
        if generation_idx + 1 < n_gens {
            // SELEÇÃO sobre a fitness escolhida, não sobre o hodômetro.
            //
            // Aqui estava `&dist_acum` fixo: eu tinha trocado só o que o relatório EXIBE
            // e deixado a seleção premiando hodômetro. As duas condições `--fitness dist`
            // e `--fitness prog` saíam idênticas ao dígito — e eu quase registrei isso
            // como "a fitness não importa". Ela nunca tinha mudado.
            genomes = evolve(&genomes, &fitness, &evo_params, gen_abs);
        }
    }

    let total_s = t_total.elapsed().as_secs_f64();
    println!();
    if !elite_reprod.is_empty() {
        let validos: Vec<f64> = elite_reprod.iter().cloned().filter(|v| v.is_finite()).collect();
        if !validos.is_empty() {
            let media = validos.iter().sum::<f64>() / validos.len() as f64;
            let mut ord = validos.clone();
            ord.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mediana = ord[ord.len() / 2];
            let piores = ord.iter().filter(|&&v| v < 0.5).count();
            println!();
            println!("=== REPRODUTIBILIDADE DO CAMPEÃO ===");
            println!("  o MESMO genoma, reavaliado na geração seguinte:");
            println!("  média {:.0}% · mediana {:.0}% do resultado anterior · n={}",
                     100.0 * media, 100.0 * mediana, validos.len());
            println!("  perdeu mais da METADE em {}/{} gerações", piores, validos.len());
            println!("  (100% = fitness é do genoma; muito abaixo = seleção premiou sorte)");
        }
    }

    println!("=== RESULTADO FINAL ({:.1}s total) ===", total_s);
    println!("  gerações: {} · seres: {} · ticks/gen: {}", n_gens, n_seres, n_ticks);
    println!();

    // ── Última geração ─────────────────────────────────────────────────────────
    if let Some(last) = hist_results.last() {
        println!("Última geração (gen {:4}):", last.generation);
        println!("  mediana: {:.0}m ({:.3} voltas)", last.dist_med, last.dist_med / lap_dist);
        println!("  máximo:  {:.0}m ({:.3} voltas)", last.dist_max, last.dist_max / lap_dist);
        println!("  ≥1 volta: {}/{}  ≥2 voltas: {}/{}", last.n_ge1v, n_seres, last.n_ge2v, n_seres);
        println!("  mortalidade: {:.1}%", last.mortality * 100.0);
        println!();
    }

    // ── Inclinação da MEDIANA com teste de janela encolhida ───────────────────
    println!("=== TRAJETÓRIA DA MEDIANA (OLS) ===");
    if hist_med.len() >= 2 {
        let (slope, stderr) = linear_slope(&hist_med);
        let sigma_ratio = if stderr > 1e-10 { slope / stderr } else { 0.0 };
        println!("  Todas {} gens: slope={:+.2}m/gen  σ={:.2}  razão={:.2}σ",
            hist_med.len(), slope, stderr, sigma_ratio);

        // Teste de janela encolhida: remove as últimas 1, 2, 3 gens e recalcula.
        // Se a inclinação some ao remover poucos pontos, era artefato de janela.
        for remover in 1..=3.min(hist_med.len().saturating_sub(2)) {
            let n_restante = hist_med.len() - remover;
            let (s2, e2) = linear_slope(&hist_med[..n_restante]);
            let sr2 = if e2 > 1e-10 { s2 / e2 } else { 0.0 };
            println!("  -{}gen ({} gens): slope={:+.2}m/gen  σ={:.2}  razão={:.2}σ",
                remover, n_restante, s2, e2, sr2);
        }
    } else {
        println!("  Dados insuficientes (< 2 gerações).");
    }

    println!();
    println!("=== TRAJETÓRIA DO MÁXIMO ===");
    if hist_max.len() >= 2 {
        let (slope, stderr) = linear_slope(&hist_max);
        let sigma_ratio = if stderr > 1e-10 { slope / stderr } else { 0.0 };
        println!("  Todas {} gens: slope={:+.2}m/gen  σ={:.2}  razão={:.2}σ",
            hist_max.len(), slope, stderr, sigma_ratio);
    }

    // ── Tabela completa das medianas ───────────────────────────────────────────
    println!();
    println!("=== TABELA COMPLETA (gen | mediana metros | voltas) ===");
    for (i, (&med, &max)) in hist_med.iter().zip(hist_max.iter()).enumerate() {
        println!("  gen {:4}: med={:7.0}m ({:.3}V) max={:7.0}m ({:.3}V)",
            gen_offset + i, med, med / lap_dist, max, max / lap_dist);
    }

    println!();

    // Imprime relatório da auditoria se ela foi ativada.
    if AUDITAR.load(Ordering::Relaxed) || audit::n_ticks_acumulados() > 0 {
        audit::auditoria_relatorio();
    }

    println!("=== FIM gpu_evolve ===");
}

