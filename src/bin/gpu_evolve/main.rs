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
mod painel;
mod pistas;
#[cfg(test)]
mod tests;

use statics::*;
use context::GpuCtx;
use types::BeingState;
use genome_io::{save_genome, load_genome, population_seed_for};
use pistas::{parse_pistas, pista_da_gen};

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

// ── Avaliação held-out (P1, docs/PLANO_EVOLUCAO.md §P1) ───────────────────────

/// Avalia a POPULAÇÃO ATUAL (mesmos genomas já upados na GPU — pesos não mudam
/// entre pistas, só a geometria muda) na pista RESERVADA, por `n_ticks` (mesmo N
/// do treino). NÃO muta `genomes` e não participa de `evolve()` — é medição pura.
///
/// # Por que a seleção nunca vê este número (item 4 da TAREFA)
///
/// `evolve()` só recebe a `fitness` de treino (chamado depois deste bloco no laço
/// principal, com o vetor calculado ANTES da avaliação held-out rodar). Se a
/// seleção enxergasse também a fitness held-out, o campeão passaria a ser
/// escolhido (também) pelo desempenho na pista reservada — e ela deixaria de medir
/// generalização, porque o próprio processo de seleção teria decorado ela também.
/// Held-out só serve para alguma coisa enquanto continuar sendo a pista que a
/// seleção NUNCA otimiza contra; por isso esta função devolve só um `f64` de
/// relatório, sem tocar em `genomes`/`EvoParams`/`evolve`.
///
/// # Simplificações deliberadas (documentadas, não esquecidas)
///
/// Reimplementa (não reusa) o núcleo de acumulação de fitness do laço principal —
/// odômetro, progresso monotônico, checkpoints em ordem, centralização — em vez de
/// extrair uma função compartilhada, porque o laço principal está entrelaçado com
/// replay/painel/dopamina/combustível/ativação, e tocar naquele bloco aumentaria o
/// risco de mudar o comportamento do A/B em andamento por acidente. Ficam de fora
/// de propósito (e não afetam a fórmula de nenhum `--fitness`):
/// - `--portal-curva`: held-out sempre usa setor UNIFORME. Comparar `razao` exige
///   a mesma contagem de setores nos dois lados; misturar contagens (43 portais no
///   treino, 24 setores no held-out) tornaria a razão incomparável.
/// - `--fuel`: held-out mede se a ROTA generaliza, não se a gestão de tanque
///   generaliza — os dois são perguntas diferentes.
/// - dopamina (`joy_tick`/`joy_streak`): não alimenta NENHUMA fórmula de fitness
///   (`dist`/`prog`/`checkpoint`/`centro`), só um canal sensorial.
/// - `--wall-cost` e vida "ganha por metro" (`--life-earn`): held-out dá vida
///   CHEIA (`n_ticks`) para medir se o corpo/cérebro navega a pista reservada, não
///   se a população sobrevive com o orçamento de fôlego calibrado para o TREINO.
///
/// Retorna a MEDIANA da fitness held-out entre os seres que terminaram VIVOS
/// (`life_left > 0`) — item 3 da TAREFA ("razão usa mediana dos vivos"). Se
/// ninguém sobreviver, cai para a mediana de TODOS (avisado via `eprintln!`) em
/// vez de indexar um vetor vazio.
fn avaliar_held_out(
    ctx: &mut GpuCtx,
    n_seres: usize,
    n_ticks: usize,
    track_width: f32,
    same_start: bool,
    modo_fitness: &str,
    held_out_seed: u64,
    kan_knots: &[[f32; universe::kan::KAN_OUT_KNOTS]; 3],
) -> f64 {
    let world_ho = VoxelWorld::new(held_out_seed, track_width);
    // Troca a geometria na GPU (grade/fluxo/alturas/heightmap) sem recriar device
    // nem pipeline — ver `GpuCtx::reload_world`. Reseta o estado recorrente (CfC)
    // para que os seres cheguem na pista reservada sem memória da pista de treino
    // desta geração — mesmo contrato de "reset entre gerações" que já existe no
    // laço principal (main(), bloco "Reseta CfC entre gerações").
    ctx.reload_world(&world_ho);
    ctx.reset_cfc(n_seres);
    for i in 0..n_seres { ctx.brain.reset_cfc(i); }

    let lap_dist = world_ho.lap_distance as f64;
    let n_wp = world_ho.waypoints_world.len();
    let mut states: Vec<BeingState> = (0..n_seres).map(|i| {
        let wi = if same_start { 0 } else { (i * n_wp / n_seres).min(n_wp - 1) };
        let wp = world_ho.waypoints_world[wi];
        let wp_next = world_ho.waypoints_world[(wi + 1).min(n_wp - 1)];
        let dx = wp_next.x - wp.x;
        let dz = wp_next.y - wp.y;
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
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
            // Vida CHEIA (ver doc da função: held-out mede rota, não orçamento de fôlego).
            life_left: n_ticks as f64,
            reverse_timer: 0.0,
            post_reverse_lockout: 0.0,
            shift_cooldown: 0.0,
            shift_up_prev: 0.0,
            shift_down_prev: 0.0,
            nm_reserva: [1.0; 4],
            joy_tick: 0.0,
            joy_streak: 0.0,
            edge_row6: 0.0,
            edge_row7: 0.0,
            phantom_d_min: 30.0,
            dist_wall_l: 40.0,
            dist_wall_r: 40.0,
            dist_wall_f: 50.0,
        }
    }).collect();

    let mut dist_acum: Vec<f64> = vec![0.0; n_seres];
    let mut prog_acum: Vec<f64> = vec![0.0; n_seres];
    let mut checkpoint: Vec<u32> = vec![0; n_seres];
    let mut prog_prev: Vec<f32> = (0..n_seres)
        .map(|i| world_ho.get_track_progress(states[i].phys.pos[0], states[i].phys.pos[2]))
        .collect();
    let mut pos_prev: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

    // Tick de aquecimento — mesmo contrato do laço de treino (JIT/compilação
    // descartada, não entra na fitness).
    let _ = ctx.tick(&mut states, n_seres, 1.0 / 64.0, 0, kan_knots);

    let dt = 1.0_f32 / 64.0;
    for tick_i in 0..n_ticks {
        if !ctx.tick(&mut states, n_seres, dt, tick_i + 1, kan_knots) {
            eprintln!("[held-out] ERRO no tick {tick_i} — avaliação abortada, usando dados parciais.");
            break;
        }
        // `ctx.tick()` já decrementa `life_left`/incrementa `ticks_alive` por conta
        // própria (mesmo caminho usado no treino) — este laço só ACUMULA fitness,
        // não mexe em vida.
        for i in 0..n_seres {
            let p1 = pos_of(&states[i].phys);
            if states[i].life_left > 0.0 {
                let dx = (p1[0] - pos_prev[i][0]) as f64;
                let dz = (p1[2] - pos_prev[i][2]) as f64;
                dist_acum[i] += (dx * dx + dz * dz).sqrt();

                let cur = world_ho.get_track_progress(p1[0], p1[2]);
                let ld = lap_dist as f32;
                let mut d = cur - prog_prev[i];
                if d < -ld * 0.5 { d += ld; } else if d > ld * 0.5 { d -= ld; }
                let avanco = d.clamp(-ld * 0.1, ld * 0.1) as f64;
                prog_acum[i] += avanco;
                prog_prev[i] = cur;

                // Centralização — só usada por `--fitness centro`; custo (9
                // comparações/ser/tick) é desprezível perto do resto do tick.
                let wps = &world_ho.waypoints_world;
                let nwp = wps.len();
                let mut melhor = states[i].wp_near;
                let mut best_d = f32::MAX;
                for off in 0..9 {
                    let idx = (states[i].wp_near + nwp + off - 4) % nwp;
                    let dxx = wps[idx].x - p1[0];
                    let dzz = wps[idx].y - p1[2];
                    let dd = dxx * dxx + dzz * dzz;
                    if dd < best_d { best_d = dd; melhor = idx; }
                }
                states[i].wp_near = melhor;
                let dist_eixo = best_d.sqrt();
                let half = (track_width * 0.5).max(1.0);
                let prox = (1.0 - dist_eixo / half).clamp(0.0, 1.0) as f64;
                if avanco > 0.0 { states[i].center_prog += avanco * prox; }

                // Setor SEMPRE uniforme (ver doc da função — held-out ignora
                // `--portal-curva` de propósito, pra manter a contagem de setores
                // comparável entre treino e held-out).
                let setor = ((cur / ld) * N_CHECKPOINTS as f32) as u32;
                let esperado = checkpoint[i] % N_CHECKPOINTS.max(1);
                if setor == esperado { checkpoint[i] += 1; }
            }
            pos_prev[i] = p1;
        }
    }

    let fitness_ho: Vec<f64> = match modo_fitness {
        "dist" => dist_acum,
        "prog" => prog_acum.iter().map(|p| p.max(0.0)).collect(),
        "centro" => (0..n_seres).map(|i| {
            checkpoint[i] as f64 * lap_dist + states[i].center_prog.max(0.0)
        }).collect(),
        _ => (0..n_seres).map(|i| {
            checkpoint[i] as f64 * lap_dist + prog_acum[i].max(0.0)
        }).collect(),
    };

    // Não deixa nenhuma amostra desta corrida vazar pro relatório de ativação da
    // PRÓXIMA geração de treino — `ctx.tick()` alimenta `l1_samples`/`l2_samples`/
    // `slow_samples`/`policy_samples`/`nan_seres` incondicionalmente quando
    // `--ativacao` está ligado, sem saber que esta corrida é held-out. Sem este
    // `clear()`, o relatório `[ativacao]`/`[lento]` da geração de treino seguinte
    // misturaria amostras da pista reservada com amostras da pista de treino.
    ctx.l1_samples.clear();
    ctx.l2_samples.clear();
    ctx.slow_samples.clear();
    ctx.policy_samples.clear();
    ctx.nan_seres.clear();
    // `regiao_l1`/`regiao_l2` (`--regioes`, src/analysis/regioes.rs) andam em lockstep
    // com `l1_samples`/`l2_samples` — mesmo risco de contaminação, mesmo conserto.
    ctx.regiao_l1.clear();
    ctx.regiao_l2.clear();

    let mediana = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let vivos: Vec<f64> = (0..n_seres)
        .filter(|&i| states[i].life_left > 0.0)
        .map(|i| fitness_ho[i])
        .collect();
    if vivos.is_empty() {
        eprintln!("[held-out] ninguém sobreviveu na pista reservada — mediana calculada \
                    sobre TODOS os seres (não só vivos), número menos confiável.");
        mediana(fitness_ho)
    } else {
        mediana(vivos)
    }
}

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
    let mut l1_vivos: usize = 0;   // 0 = todos os 96 (sem ablacao)
    let mut retomar_mix: f32 = 0.85;   // fracao do campeao preservada nos filhos
    let mut congela_l2_cegas: bool = false;
    let mut mascara_regioes: bool = false;
    let mut tau_semeado: bool = false;
    let mut observar_path: Option<String> = None;
    let mut trocar_ao_completar: usize = 0;  // 0 = rotacao por geracao (comportamento atual)
    /// P1.8, achado "currículo avança com UM ser rastejando promove a turma"
    /// (`--trocar-ao-completar 1` deixava um único ser lento fechar a lição pra
    /// população inteira). Piso: promoção exige `v1` (voltas REAIS, não hodômetro)
    /// cobrindo pelo menos 25% da população, OU (se o usuário aceitar um `N`
    /// menor) a mediana de velocidade da geração estar acima de um piso mínimo —
    /// nenhuma das duas rotas sozinha valida "aprendeu", mas as duas juntas
    /// cobrem tanto amplitude (quantos aprenderam) quanto ritmo (não é rastejo).
    /// `--curriculo-veloc-min <m/s>` deixa o piso de velocidade ajustável — 5,0
    /// m/s (18 km/h) é ~2× acima do pior "rastejar" já medido neste projeto
    /// (1,48-2,96 m/s com vida fixa, ver comentário de `WALL_LIFE_DRAIN` acima) e
    /// bem abaixo do ritmo de corrida medido (79 km/h ≈ 22 m/s), então não deveria
    /// travar progresso real de currículo — é ponto de partida, não medição final.
    let mut curriculo_veloc_min: f64 = 5.0;
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
    // P1.7 (docs/PLANO_EVOLUCAO.md) — cérebro assíncrono: memória por neurônio,
    // período/limiar de contribuição, ativação heterogênea. Os três mecanismos
    // ficam LIGADOS por padrão (genoma nasce neutro — ver `Genome::random` — e só
    // diverge por mutação, o mesmo precedente do banco de τ). As quatro flags
    // abaixo são o braço de controle: cada uma CONGELA seu campo no valor de
    // nascença pelo resto da corrida E manda o shader ignorá-lo mesmo que um
    // `--retomar` anterior já tivesse mutado algo (ver `BrainGpu::mech_off_mask`).
    let mut sem_memoria_neuronal = false;
    let mut sem_periodo_neuronal = false;
    let mut sem_limiar_neuronal = false;
    let mut sem_ativacao_heterogenea = false;
    /// `--centro-simetrico`: recuar SUBTRAI da fitness de centralização.
    let mut centro_simetrico = false;
    /// `--out-beta`: força da fadiga de saída. Padrão = o histórico.
    let mut out_beta: f32 = universe::gpu::brain::OUT_BETA_PADRAO;
    /// `--ganho-saida`: ganho aplicado a l2_raw/bypass na ETAPA 2 antes do tanh
    /// (P1.5, `docs/PLANO_EVOLUCAO.md`). Padrão = a constante 2.0 histórica —
    /// o A/B pendente compara 2.0 vs 1.0 vs 0.5 contra a saturação de L2
    /// (48/48 neurônios ativos em 97,4% dos ticks).
    let mut ganho_saida: f32 = universe::gpu::brain::OUT_GAIN_PADRAO;
    /// `--norm-l2-por-desvio` / `--sem-norm-l2-por-desvio`: segunda alavanca de
    /// P1.5. LIGADA por padrão desde a auditoria P1.8 (docs/PLANO_EVOLUCAO.md,
    /// achado "L2 tem que funcionar") — normaliza l2_raw pelo desvio-padrão dos
    /// 48 valores do MESMO ser antes do ganho/tanh. Ataca a ESCALA da entrada de
    /// L2 (causa raiz medida: σ(l2_raw) ≈ 2,9-5,9 com σ_pesos=0,30 e out_gain=2,0,
    /// suficiente pra saturar tanh sozinho) em vez do ganho de saída, que a mesma
    /// auditoria mostrou não resolver de forma robusta em nenhum valor plausível
    /// (2.0/1.0/0.5). `--sem-norm-l2-por-desvio` é o braço de controle (comportamento
    /// histórico, idêntico ao pré-P1.8) para quem precisar comparar.
    let mut norm_l2_por_desvio = true;
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
    // `--regioes`: cruza as amostras de `--ativacao` com a curvatura local da pista
    // (src/analysis/regioes.rs) — reta vs curva por neurônio. Requer `--ativacao`
    // ligada junto (a coleta em `ctx.tick()` só roda quando ela está ligada); sem
    // `--ativacao`, `--regioes` sozinha não imprime nada (vetores ficam vazios).
    let mut regioes_on = false;
    // `--painel [--painel-porta N]`: painel HTTP tempo-real (E5). Sem a flag, nenhuma
    // thread é lançada — zero overhead (REFACTOR_PLAN.md §3.7).
    let mut painel_on = false;
    let mut painel_porta: u16 = 3200;
    /// `--pistas "9001,9002,..."`: conjunto de pistas de TREINO, rotacionadas
    /// deterministicamente por geração (`pista_da_gen`, gen_abs % len). Vazio
    /// (padrão) = comportamento anterior a esta flag: pista 9001 fixa — o A/B em
    /// andamento (campeão treinado só nela, `--same-start`) continua reproduzível
    /// sem tocar em nada.
    let mut pistas: Vec<u64> = Vec::new();
    /// `--pista-held-out <seed>`: pista RESERVADA, nunca sorteada por `pista_da_gen`
    /// pro treino — só para MEDIR generalização (ver bloco `[generalizacao]`).
    /// `None` (padrão) desliga a medição inteira: nenhum tick extra roda.
    let mut pista_held_out: Option<u64> = None;
    /// `--held-out-cada <K>`: mede held-out a cada K gerações (padrão 5). Só lido
    /// quando `pista_held_out` está setada.
    let mut held_out_cada: usize = 5;
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
                // Curva de capacidade: quantos dos 96 neuronios de L1 ficam vivos.
                // O PR mede 6,7 dimensoes em uso; esta flag testa se isso se traduz em
                // metros — cortar ate onde o desempenho aguentar.
                "--l1-vivos" => { i += 1; if i < args.len() { l1_vivos = args[i].parse().unwrap_or(0); } }
                "--retomar-mix" => { i += 1; if i < args.len() { retomar_mix = args[i].parse().unwrap_or(0.85); } }
                "--trocar-ao-completar" => { i += 1; if i < args.len() { trocar_ao_completar = args[i].parse().unwrap_or(1); } }
                // P1.8: piso de velocidade mediana para a rota alternativa de promoção
                // do currículo (ver doc de `curriculo_veloc_min` acima).
                "--curriculo-veloc-min" => { i += 1; if i < args.len() { curriculo_veloc_min = args[i].parse().unwrap_or(curriculo_veloc_min); } }
                // Banco de τ log-uniforme em [25 ms, 4 s] em vez do τ = 2,0 s uniforme.
                // Sem a flag o comportamento é o histórico — é o braço de controle.
                "--tau-hetero" => { tau_hetero = true; }
                // P0.5: inicialização espectral de L1 vem LIGADA por padrão (genoma
                // nasce com ganho espectral ≈ 1.0 em vez do ~7.0 cru medido com
                // σ=0.30 i.i.d. — ver `gpu::evolution::Genome::random`). Esta flag
                // DESLIGA para rodar o braço de controle do A/B.
                "--sem-init-espectral" => {
                    universe::gpu::evolution::SPECTRAL_INIT_ON.store(false, Ordering::Relaxed);
                }
                // FASE SENSORES (degrau 1): liga a evolução de FOV/alcances/lookaheads de
                // curvatura (ver doc de `SENSORS_EVOLVE_ON`). Padrão DESLIGADA — sem esta
                // flag, todo genoma nasce e permanece em `SENSOR_DEFAULTS` (os hardcodes de
                // hoje), protegendo qualquer bateria em curso que não passe a flag nova.
                "--sensores-evolutivos" => {
                    universe::gpu::evolution::SENSORS_EVOLVE_ON.store(true, Ordering::Relaxed);
                }
                // Força de volta ao braço de controle (já é o padrão) — nome literal pedido
                // pela tarefa. Vence se vier DEPOIS de `--sensores-evolutivos` na linha de
                // comando, igual `--sem-lento`/`--sem-init-espectral` para outros experimentos.
                "--sensores-fixos" => {
                    universe::gpu::evolution::SENSORS_EVOLVE_ON.store(false, Ordering::Relaxed);
                }
                // P1.7 (docs/PLANO_EVOLUCAO.md) — braço de controle do cérebro
                // assíncrono, um por mecanismo. Cada um CONGELA a mutação do campo
                // (atomic em `gpu::evolution`) E força o shader a ignorar o que o
                // genoma carregar (`BrainGpu::*_off`) — ver a doc das variáveis
                // `sem_*` acima. Sem nenhuma destas flags (padrão), os três
                // mecanismos mutam livremente a partir do neutro de nascença.
                "--sem-memoria-neuronal" => {
                    sem_memoria_neuronal = true;
                    universe::gpu::evolution::NEURON_MEM_MUTATE_ON.store(false, Ordering::Relaxed);
                }
                "--sem-periodo-neuronal" => {
                    sem_periodo_neuronal = true;
                    universe::gpu::evolution::NEURON_PERIOD_MUTATE_ON.store(false, Ordering::Relaxed);
                }
                "--sem-limiar-neuronal" => {
                    sem_limiar_neuronal = true;
                    universe::gpu::evolution::NEURON_THRESHOLD_MUTATE_ON.store(false, Ordering::Relaxed);
                }
                "--sem-ativacao-heterogenea" => {
                    sem_ativacao_heterogenea = true;
                    universe::gpu::evolution::NEURON_ACTIVATION_MUTATE_ON.store(false, Ordering::Relaxed);
                }
                // Cobra o recuo na fitness `centro` (ver a acumulação de `center_prog`).
                "--centro-simetrico" => { centro_simetrico = true; }
                // Força da fadiga de SAÍDA. 0,85 (padrão) faz comando sustentado sair a
                // 15% da intenção em ~1 s; 0,0 desliga. Ver `BrainParams::out_beta`.
                "--out-beta"  => { i += 1; if i < args.len() { out_beta = args[i].parse().unwrap_or(out_beta); } }
                // P1.5: ganho de saída (era a constante 2.0 hardcoded na ETAPA 2).
                // Ver `BrainParams::out_gain` e o comentário de `ganho_saida` acima.
                "--ganho-saida" => { i += 1; if i < args.len() { ganho_saida = args[i].parse().unwrap_or(ganho_saida); } }
                // P1.5/P1.8: normaliza l2_raw pelo desvio-padrão dos 48 valores do ser
                // antes do ganho/tanh. LIGADA por padrão desde P1.8 — esta flag vira
                // no-op explícito (mantida para não quebrar scripts existentes).
                "--norm-l2-por-desvio" => { norm_l2_por_desvio = true; }
                // Braço de controle: desliga a normalização e volta ao comportamento
                // histórico pré-P1.8 (l2_std fixo em 1.0, divisão é no-op no shader).
                "--sem-norm-l2-por-desvio" => { norm_l2_por_desvio = false; }
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
                // FÍSICA 1: vida/tick queimada por girar acima de 90% do RPM (0=off).
                // E-REG1: λ do custo de fiação por distância (0=off). Ver statics.rs.
                "--fiacao-cauda" => { i += 1; if i < args.len() {
                    FIACAO_CAUDA.store(args[i].parse().unwrap_or(0), Ordering::Relaxed);
                } }
                "--custo-fiacao" => { i += 1; if i < args.len() {
                    CUSTO_FIACAO.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                } }
                "--custo-rpm" => { i += 1; if i < args.len() {
                    CUSTO_RPM.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                } }
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
                // Remove os buracos intencionais da pista (world.rs:141). Existia so no
                // caminho Bevy (main.rs:50); exposto aqui porque a bateria pos-refatoracao
                // mostrou a POPULACAO estacionada em 1.076 m por 20 geracoes — 11 m antes
                // do campo de buracos (1.087-1.092 m). Este A/B e o que separa "nao
                // aprendem a desviar" de "os buracos sao a segunda parede".
                "--sem-buracos" => { universe::world::NO_HOLES.store(true, Ordering::Relaxed); }
                // Câmbio: travado (histórico, teto 21,5 m/s), auto (RPM, paridade Bevy)
                // ou neural (auto + intent[3]/[4]). Ver CAMBIO_MODO em statics.rs.
                "--cambio-curriculo" => { CAMBIO_CURRICULO.store(true, Ordering::Relaxed); }
                "--cambio-teto" => { i += 1; if i < args.len() {
                    CAMBIO_TETO.store(args[i].parse().unwrap_or(6).clamp(2, 6), Ordering::Relaxed);
                } }
                "--cambio"    => { i += 1; if i < args.len() {
                    let m = match args[i].as_str() {
                        "travado" => 0u32, "auto" => 1, "neural" => 2,
                        outro => { eprintln!("--cambio '{outro}' inválido (travado|auto|neural); usando travado"); 0 }
                    };
                    CAMBIO_MODO.store(m, Ordering::Relaxed);
                } }
                // Congela a mutação das linhas 5..48 de L2 (saídas que nenhum pipeline
                // lê — "feature neurons" sem consumidor; 4.128 pesos de busca inerte).
                "--l2-congelar-cegas" => { congela_l2_cegas = true; }
                // M3: regiões DIRIGIDAS (teto de referência) — ver evolution.rs::mascara_permite.
                "--mascara-regioes" => { mascara_regioes = true; }
                // E-REG2 (M2): semeia τ por BLOCO (mesmos 4 blocos do M3) — reflexo
                // curto na parede/estrada, longo no associativo. Função segue tempo?
                "--tau-semeado" => { tau_semeado = true; }
                // M4: maestro de regiões (lento→gates com reserva). Ver statics.rs.
                "--neuromod-regioes" => { NEUROMOD_REGIOES.store(true, Ordering::Relaxed); }
                // Laboratório de observação: grava a cadeia neural do ser 0 por tick
                // (394 f32/linha, ver context.rs). Usar com --retomar --gens 1.
                "--observar"  => { i += 1; if i < args.len() { observar_path = Some(args[i].clone()); } }
                "--portal-curva" => { portal_curva = true; }
                "--slow-steer" => { i += 1; if i < args.len() { slow_steer_auth = args[i].parse().unwrap_or(0.20); } }
                "--refuel-laps"=>{ i += 1; if i < args.len() { refuel_laps = args[i].parse().unwrap_or(1).max(1); } }
                "--life-cap"  => { i += 1; if i < args.len() { life_cap_arg  = args[i].parse().unwrap_or(0); } }
                "--n-elite"   => { i += 1; if i < args.len() { evo_n_elite   = args[i].parse().unwrap_or(3); } }
                "--tourney"   => { i += 1; if i < args.len() { evo_tourney   = args[i].parse().unwrap_or(3); } }
                "--gen-offset"=> { i += 1; if i < args.len() { gen_offset    = args[i].parse().unwrap_or(0); } }
                "--ativacao"   => { ativacao_on = true; }
                "--ativacao-n" => { i += 1; if i < args.len() { ativacao_n = args[i].parse().unwrap_or(64).max(1); } }
                "--regioes"    => { regioes_on = true; }
                "--painel"       => { painel_on = true; }
                "--painel-porta" => { i += 1; if i < args.len() { painel_porta = args[i].parse().unwrap_or(3200); painel_on = true; } }
                "--pistas"        => { i += 1; if i < args.len() { pistas = parse_pistas(&args[i]); } }
                "--pista-held-out" => { i += 1; if i < args.len() { pista_held_out = args[i].parse().ok(); } }
                "--held-out-cada" => { i += 1; if i < args.len() { held_out_cada = args[i].parse().unwrap_or(5).max(1); } }
                "--retomar"   => { retomar = true; }
                "--out"       => { i += 1; if i < args.len() { out_dir       = args[i].clone(); } }
                _ => {}
            }
            i += 1;
        }
    }

    let inicio = if retomar { "retomando de checkpoint" } else { "pesos ALEATÓRIOS (σ=0.30, N(0,σ))" };
    let spectral_init_on = universe::gpu::evolution::SPECTRAL_INIT_ON.load(Ordering::Relaxed);
    println!("=== gpu_evolve: {} seres · {} ticks/gen · {} gens ===", n_seres, n_ticks, n_gens);
    println!("  Inicio: {}", inicio);
    println!("  Init espectral L1 (P0.5): {}", if spectral_init_on {
        "LIGADA — L1 nasce com ganho espectral ≈ 1.0 (alvo em thresholds::SPECTRAL_INIT_TARGET)"
    } else {
        "DESLIGADA (--sem-init-espectral) — L1 nasce cru, σ=0.30 i.i.d. sem correção (braço de controle)"
    });
    let sensors_evolve_on = universe::gpu::evolution::SENSORS_EVOLVE_ON.load(Ordering::Relaxed);
    println!("  Sensores (FASE SENSORES, degrau 1): {}", if sensors_evolve_on {
        "EVOLUINDO (--sensores-evolutivos) — FOV/alcance-parede/alcance-lateral/curvatura heterogêneos e mutáveis"
    } else {
        "FIXOS (padrão / --sensores-fixos) — todo genoma em SENSOR_DEFAULTS, os hardcodes de hoje"
    });
    // P1.7 — cérebro assíncrono: relatório de quais dos três mecanismos estão
    // mutando nesta corrida. Genoma sempre nasce neutro (gen 0 idêntica ao
    // cérebro de hoje); o que muda entre "ligado"/"--sem-*" é se a mutação afasta
    // o campo do neutro ao longo das gerações.
    println!("  Cérebro assíncrono (P1.7):");
    println!("    memória por neurônio:  {}", if sem_memoria_neuronal {
        "CONGELADA (--sem-memoria-neuronal) — alfa preso em 0.0, braço de controle"
    } else { "mutando (padrão) — h[n]=h[n-1]·α+x[n]·(1-α), α∈[0,0.95] a partir de 0.0" });
    println!("    período/limiar (L1):   {}{}", if sem_periodo_neuronal {
        "período CONGELADO (--sem-periodo-neuronal) "
    } else { "período mutando (padrão) " }, if sem_limiar_neuronal {
        "· limiar CONGELADO (--sem-limiar-neuronal)"
    } else { "· limiar mutando (padrão)" });
    println!("    ativação heterogênea:  {}", if sem_ativacao_heterogenea {
        "CONGELADA (--sem-ativacao-heterogenea) — todos em tanh, braço de controle"
    } else { "mutando (padrão) — tanh/ReLU/linear/leaky-ReLU, taxa 0,2%/neurônio/geração" });
    println!("  Evolução: sigma={:.3}  p_mutate={:.3}  n_elite={}  tourney={}", evo_sigma, evo_p_mutate, evo_n_elite, evo_tourney);
    println!("  SEM muletas motoras (escape_ticks, curvature_governor)");
    println!("  Fitness: distância contínua em metros (sem degrau de volta inteira)");
    println!();

    // ── Rotação de pistas (P1, docs/PLANO_EVOLUCAO.md) ────────────────────────
    // Aviso não-fatal: se a pista reservada também estiver no conjunto de treino,
    // ela deixa de medir generalização — o campeão veria a "reservada" durante o
    // treino normal, então nunca é surpreendido por ela no bloco [generalizacao].
    if let Some(ho) = pista_held_out {
        if pistas.contains(&ho) {
            eprintln!(
                "[AVISO] --pista-held-out {} também está em --pistas — ela não é mais \
                 reservada, a medição de generalização fica inválida.", ho
            );
        }
    }

    // ── Gate de passabilidade (P1.8) ────────────────────────────────────────────
    // Achado "pista_check nunca é chamado por gpu_evolve — não existe gate contra
    // pista com defeito": sementes procedurais podiam ter eixo cortando parede ou
    // estrangulamento e nada impedia uma bateria inteira de treinar contra um mapa
    // impossível (quatro controladores independentes já pararam no mesmo ponto por
    // isso). Valida TODAS as seeds de `--pistas`/`--pista-held-out` ANTES de subir
    // GPU/gastar uma geração sequer — mesma checagem de `pista_check.rs`
    // (`VoxelWorld::validar_pista`), sem precisar rodar o binário à parte.
    if !pistas.is_empty() || pista_held_out.is_some() {
        let mut seeds_a_checar: Vec<u64> = pistas.clone();
        if let Some(ho) = pista_held_out {
            if !seeds_a_checar.contains(&ho) { seeds_a_checar.push(ho); }
        }
        let mut alguma_falhou = false;
        for &s in &seeds_a_checar {
            let w = { let _q = universe::world::silence_map_diag(); VoxelWorld::new(s, track_width) };
            let ate = w.lap_distance.max(1.0);
            let defeitos = w.validar_pista(track_width, ate);
            if !defeitos.is_empty() {
                alguma_falhou = true;
                eprintln!(
                    "[pista_check] seed {s}: {} defeito(s) em {ate:.0}m de eixo — {}{}",
                    defeitos.len(),
                    defeitos.iter().take(3).cloned().collect::<Vec<_>>().join(" | "),
                    if defeitos.len() > 3 { " | ..." } else { "" }
                );
            } else {
                println!("[pista_check] seed {s}: OK ({ate:.0}m de eixo examinados, sem defeito)");
            }
        }
        if alguma_falhou {
            eprintln!(
                "\nERRO: uma ou mais pistas de --pistas/--pista-held-out têm defeito \
                 geométrico (eixo fora do asfalto ou estrangulamento) — nenhum controlador \
                 venceria essa pista (achado P1.8). Rode `pista_check --seed <N> --width {track_width}` \
                 para o detalhe completo e ajuste --pistas/--width antes de treinar."
            );
            std::process::exit(1);
        }
    }

    // ── Mundo ─────────────────────────────────────────────────────────────────
    // Seed = a pista de TREINO da primeira geração que este processo vai rodar
    // (`gen_offset` != 0 ao retomar). Sem `--pistas` (lista vazia), `pista_da_gen`
    // devolve 9001 para qualquer geração — mesmo hardcode literal de antes desta
    // mudança, então o A/B em andamento não é afetado.
    let t0 = Instant::now();
    let mut world_seed_atual = pista_da_gen(&pistas, gen_offset);
    // Índice do currículo: anda por MÉRITO (volta completada), não por geração.
    let mut idx_curriculo: usize = gen_offset;
    let mut world = VoxelWorld::new(world_seed_atual, track_width);
    let mut lap_dist = world.lap_distance as f64;
    println!("mundo: {}×{} · {} wp · volta={:.0}m  [{:.0}ms] · pista inicial seed={}",
        world.size, world.size, world.waypoints_world.len(), lap_dist,
        t0.elapsed().as_secs_f64() * 1000.0, world_seed_atual);
    if !pistas.is_empty() {
        println!("  [pistas] rotação LIGADA: {:?} (gen % {})", pistas, pistas.len());
    }
    if let Some(ho) = pista_held_out {
        println!("  [pistas] held-out seed={} a cada {} gerações (medição pura, NUNCA entra na seleção)",
                 ho, held_out_cada);
    }
    println!();

    // ── Painel HTTP tempo-real (E5, `--painel`) ────────────────────────────────
    // Lançado ANTES do GPU subir de propósito: `GpuCtx::new` pode levar segundos, e o
    // traçado (`set_track`) já fica disponível em `/api/status` enquanto isso acontece.
    let painel: Option<painel::PainelHandle> = if painel_on {
        let p = painel::PainelHandle::launch(painel_porta);
        p.set_track(&world.waypoints_world);
        Some(p)
    } else {
        None
    };

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
    if CAMBIO_CURRICULO.load(Ordering::Relaxed) {
        // O valor de --cambio-teto vira o MÁXIMO; o teto vivo começa em 2 (≡ travado).
        CAMBIO_TETO_MAX.store(CAMBIO_TETO.load(Ordering::Relaxed), Ordering::Relaxed);
        CAMBIO_TETO.store(2, Ordering::Relaxed);
        println!("  [cambio] currículo de teto LIGADO: começa na marcha 2, sobe por mérito até {}",
                 CAMBIO_TETO_MAX.load(Ordering::Relaxed));
    }
    if let Some(ref path) = observar_path {
        match std::fs::File::create(path) {
            Ok(f) => {
                ctx.obs_writer = Some(std::io::BufWriter::new(f));
                println!("  [observar] gravando cadeia neural do ser 0 em {path} (394 f32/tick)");
            }
            Err(e) => { eprintln!("ERRO --observar {path}: {e}"); std::process::exit(1); }
        }
    }
    // Fitness do melhor da geração anterior — o mesmo genoma reaparece sem mutação no
    // slot 0 da geração seguinte, então a comparação é do genoma consigo mesmo.
    let mut elite_fitness_anterior: Option<f64> = None;
    // P1.8, achado "métrica de reprodução do elite não corrige quando a pista
    // muda entre as duas gerações comparadas" — guarda em QUAL pista a fitness
    // anterior foi medida; só compara/empurra pra `elite_reprod` quando a pista
    // desta geração é a MESMA.
    let mut elite_fitness_anterior_pista: Option<u64> = None;
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
    ctx.brain.out_gain = ganho_saida;
    ctx.brain.norm_l2_by_std = norm_l2_por_desvio;
    // P1.7 — mesmo bit que congela a mutação em `evolve()` (armado acima, no parsing
    // das flags) TAMBÉM desliga a LEITURA do campo no shader — ver `mech_off_mask`
    // em `BrainGpu`/`brain.wgsl`. Redundante quando o genoma nunca mutou (o campo já
    // está neutro), mas necessário quando `--retomar` carrega um genoma que já tinha
    // mutado ANTES da flag ser passada: sem isto, `--sem-X` deixaria de ser o braço
    // de controle de verdade a partir do segundo `--retomar`.
    ctx.brain.mem_off = sem_memoria_neuronal;
    ctx.brain.period_off = sem_periodo_neuronal;
    ctx.brain.threshold_off = sem_limiar_neuronal;
    ctx.brain.activation_off = sem_ativacao_heterogenea;
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
    if (ganho_saida - universe::gpu::brain::OUT_GAIN_PADRAO).abs() > 1e-6 {
        println!("[P1.5] ganho de saída: out_gain={ganho_saida:.2} (padrão {:.2}) — ETAPA 2 \
                  (l2_raw e bypass) escalada antes da fadiga/tanh",
                 universe::gpu::brain::OUT_GAIN_PADRAO);
    }
    if norm_l2_por_desvio {
        println!("[P1.5/P1.8] normalização de L2 por desvio-padrão LIGADA (padrão desde P1.8) \
                  — l2_raw dividido pelo desvio dos 48 valores do ser antes do ganho/tanh");
    } else {
        println!("[P1.5/P1.8] normalização de L2 por desvio-padrão DESLIGADA (--sem-norm-l2-por-desvio) \
                  — braço de controle, comportamento histórico pré-P1.8 (l2_std=1.0, saturação de L2 \
                  não corrigida)");
    }
    println!("GpuCtx pronto  [{:.0}ms]", t1.elapsed().as_secs_f64() * 1000.0);

    // ── Genomas iniciais ───────────────────────────────────────────────────────
    // P1.8, achado "--retomar + --tau-hetero descarta o τ evoluído do campeão":
    // `true` só quando `genomes[0]` é o clone EXATO de um `.bin` carregado do disco
    // (linha "elite exato" abaixo) — usado mais abaixo pra NÃO sobrescrever o τ
    // desse slot específico com `randomize_tau` (τ muta incondicionalmente em
    // `evolve()`, então o τ de um campeão de várias gerações já passou por
    // seleção real; os demais slots, mesmo vindos do campeão, são filhos
    // ATENUADOS — não "exatos" — então ganhar τ heterogêneo neles é consistente
    // com o resto do design de `--retomar-mix`).
    let mut elite_carregado_do_checkpoint = false;
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
                elite_carregado_do_checkpoint = true;
                // Semente a população a partir do melhor genoma + variações.
                let mut pop = Vec::with_capacity(n_seres);
                pop.push(best.clone());  // elite exato
                // ── COMO OS FILHOS DO CAMPEÃO NASCEM ────────────────────────────
                //
                // Leonardo: *"faça uma mutação superleve nos outros carros, nada de 70/30,
                // mais algo de 85/15 para preservar mais o campeão — só que sem pesos
                // aleatórios, pode ser pesos zerados."*
                //
                // Duas mudanças, e a segunda importa mais que a primeira:
                //
                // 1. A proporção passa a ser parâmetro (`--retomar-mix`, padrão 0,85). O
                //    70/30 anterior custava caro: o campeão recarregado rendia ~1.076 m
                //    contra 4.223 m do original, o que inviabilizava reavaliar o campeão
                //    puro e contaminou a curva de ablação desta sessão.
                //
                // 2. O que entra na mistura deixa de ser ruído e passa a ser ZERO. A
                //    diferença é de natureza, não de grau: misturar com um genoma aleatório
                //    empurra cada peso na direção de um valor SORTEADO — é adicionar um
                //    cérebro estranho ao do pai. Misturar com zero apenas ATENUA o peso
                //    (`w · 0,85`), preservando a direção do vetor e reduzindo o ganho.
                //    O filho fica sendo o mesmo cérebro, mais tímido — nada é inventado.
                //    Cosseno com o pai permanece 1,0 por construção; só a norma cai.
                //
                // A diversidade da população passa a vir da mutação gaussiana normal ao
                // longo das gerações, não de um chute inicial que destrói o que foi ganho.
                let mix = retomar_mix.clamp(0.0, 1.0);
                println!(
                    "  filhos do campeão: {:.0}% do pai + {:.0}% zero (atenuação, sem ruído)",
                    mix * 100.0, (1.0 - mix) * 100.0);
                while pop.len() < n_seres {
                    let mut filho = best.clone();
                    for w in filho.l1.iter_mut() { *w *= mix; }
                    for w in filho.l2.iter_mut() { *w *= mix; }
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
        // Pula o índice 0 quando ele é o "elite exato" vindo de checkpoint —
        // sobrescrever o τ dele quebraria a garantia de clone exato e descartaria
        // τ que já passou por seleção real em gerações anteriores (ver doc de
        // `elite_carregado_do_checkpoint` acima).
        let start = if elite_carregado_do_checkpoint { 1 } else { 0 };
        if elite_carregado_do_checkpoint {
            println!("  [tau-hetero] preservando τ evoluído do elite exato (slot 0) — \
                      banco novo só nos demais {} slots", n_seres.saturating_sub(1));
        }
        for (i, g) in genomes.iter_mut().enumerate().skip(start) {
            g.randomize_tau(population_seed_for(pop_seed, i).wrapping_add(0x7A11_0000));
        }
    }

    // Parâmetros evolutivos: configuráveis via CLI, default conservador.
    if tau_semeado {
        // E-REG2: priors de τ por bloco (log-uniforme na faixa; d = 1/τ):
        //   bloco 0 visão: livre (0,5-4 s, faixa atual) · bloco 1 parede/estrada:
        //   REFLEXO 0,3-0,8 s · bloco 2 corpo: 1-4 s · bloco 3 associativo: 4-15 s.
        // Sobrescreve τ herdado em TODA a população (é a intervenção; a mutação
        // segue refinando a partir daí).
        for (gi, g) in genomes.iter_mut().enumerate() {
            for j in 0..universe::config::iara::brain::L1_NEURONS {
                let (lo, hi): (f64, f64) = match j / 24 {
                    0 => (0.5, 4.0),
                    1 => (0.3, 0.8),
                    2 => (1.0, 4.0),
                    _ => (4.0, 15.0),
                };
                let u = {
                    let mut x = (gi as u64 * 96 + j as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(pop_seed);
                    x ^= x >> 33; x = x.wrapping_mul(0xFF51AFD7ED558CCD); x ^= x >> 33;
                    (x % 10_000) as f64 / 10_000.0
                };
                let tau = lo * (hi / lo).powf(u);
                g.tau[j] = (1.0 / tau) as f32;
            }
        }
        println!("  [tau-semeado] E-REG2: τ por bloco (visão 0,5-4s · reflexo 0,3-0,8s · corpo 1-4s · associativo 4-15s)");
    }
    if mascara_regioes {
        for g in genomes.iter_mut() { universe::gpu::evolution::aplicar_mascara_regioes(g); }
        println!("  [mascara] regiões DIRIGIDAS ativas (4 blocos de 24: visão/parede/corpo/associativo)");
    }
    let evo_params = EvoParams {
        p_mutate:    evo_p_mutate,
        sigma:       evo_sigma,
        n_elite:     evo_n_elite,
        tournament_k: evo_tourney,
        congela_l2_cegas,
        mascara_regioes,
        // 5 linhas lidas só quando o câmbio neural dá leitor ao intent[3]/[4];
        // travado/auto: 3 (throttle/steer/brake).
        l2_linhas_lidas: if CAMBIO_MODO.load(Ordering::Relaxed) == 2 { 5 } else { 3 },
    };

    // ── Histórico de medianas (para inclinação ao final) ──────────────────────
    let mut hist_med:  Vec<f64> = Vec::with_capacity(n_gens);
    // Volta da pista em que CADA geração rodou — a tabela final normalizava tudo pela
    // última pista carregada e imprimia V errada em bateria com rotação de currículo
    // (medido: gen28 rodou 9001/2928m, tabela dividiu por 3691m e mostrou 1.256V).
    let mut hist_lap:  Vec<f64> = Vec::with_capacity(n_gens);
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

        // ── Rotação de pistas (P1) ──────────────────────────────────────────────
        // `seed_gen` é a pista de TREINO desta geração. Sem `--pistas` (lista
        // vazia), `pista_da_gen` sempre devolve 9001 — `seed_gen == world_seed_atual`
        // em toda geração, o `if` abaixo nunca dispara, e o comportamento fica
        // idêntico ao binário de antes desta mudança (mundo criado uma vez, nunca
        // recarregado). `world_seed_atual` também é o que a avaliação held-out
        // (mais abaixo) deixa "sujo" ao trocar de pista por conta própria — este
        // `if` é o que traz o treino de volta no início da PRÓXIMA geração, sem
        // precisar de nenhum código de "restaurar" no bloco held-out em si.
        // ── CURRÍCULO: avança a pista quando o circuito for DOMINADO ─────────
        //
        // Leonardo: *"pegue o campeão e rode em pistas diferentes; quando ele completar o
        // circuito, troque de pista."*
        //
        // A rotação por geração (`gen % n`) troca a pista no relógio, independente de o ser
        // estar aprendendo ou apanhando. Com `--trocar-ao-completar N`, o índice do
        // currículo só avança quando pelo menos N seres completam uma volta: enquanto a
        // pista não for vencida, ela continua; vencida, vem a próxima. É a diferença entre
        // trocar de assunto no relógio e trocar quando a lição foi aprendida.
        //
        // O índice é separado de `gen_abs` justamente porque um deles anda sozinho e o
        // outro só anda por mérito.
        let seed_gen = pista_da_gen(&pistas, if trocar_ao_completar > 0 { idx_curriculo } else { gen_abs });
        if seed_gen != world_seed_atual {
            world = VoxelWorld::new(seed_gen, track_width);
            ctx.reload_world(&world);
            world_seed_atual = seed_gen;
            lap_dist = world.lap_distance as f64;
        }
        println!("  [pista] gen {:4} usa seed={} ({} wp, volta={:.0}m)",
                 gen_abs, seed_gen, world.waypoints_world.len(), lap_dist);

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
        // `seed_gen` (não mais 9001 hardcoded): com rotação ligada, gravar sempre
        // 9001 faria o replay MENTIR sobre qual pista produziu aquele arquivo —
        // risco item do mapa P1. Sem `--pistas`, `seed_gen == 9001` sempre, e este
        // campo do arquivo fica idêntico ao que já era.
        let mut replay_writer = universe::replay::ReplayWriter::create(
            &replay_path, universe::replay::SOURCE_CORTEX, n_grav as u8, 64.0, seed_gen,
        ).ok();
        let t_gen = Instant::now();

        // ── Aplica genomas nos engines ─────────────────────────────────────────
        let mut engines: Vec<SovereignEngine> = (0..n_seres).map(|_| SovereignEngine::new()).collect();
        for (i, (eng, genome)) in engines.iter_mut().zip(genomes.iter()).enumerate() {
            if l1_vivos > 0 {
                // Ablacao aplicada numa COPIA: o genoma herdado fica intacto, so o corpo
                // avaliado perde os neuronios. Assim a curva mede o efeito de ter menos
                // neuronios ATIVOS, sem mutilar a heranca ao longo das geracoes.
                let mut g = genome.clone();
                g.ablar_l1_acima_de(l1_vivos);
                g.apply_to_engine(eng);
            } else {
                genome.apply_to_engine(eng);
            }
        }
        // E-REG1: custo de fiação por ser, recalculado por geração (pesos fixos na
        // vida). dist circular entre pos do canal (c/192) e do neurônio (j/96);
        // custo = λ × média ponderada por |w| × 2 ∈ [0, λ] vida/tick.
        {
            let lam = CUSTO_FIACAO.load(Ordering::Relaxed);
            if lam > 0.0 {
                let cauda = FIACAO_CAUDA.load(Ordering::Relaxed) as usize;
                ctx.fiacao_custo = genomes.iter().map(|g| {
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for j in 0..universe::config::iara::brain::L1_NEURONS {
                        let pj = j as f64 / 96.0;
                        let linha: Vec<(f64, f64)> = (0..universe::config::iara::brain::SENSORY_WAVE_SIZE).map(|c| {
                            let w = g.l1[j * universe::config::iara::brain::SENSORY_WAVE_SIZE + c].abs() as f64;
                            let pc = c as f64 / 192.0;
                            let d = (pj - pc).abs();
                            (w, d.min(1.0 - d))
                        }).collect();
                        // E-REG1b: com --fiacao-cauda N, só as N conexões mais FORTES do
                        // neurônio pagam fio — é onde os genomas diferem (ver statics.rs).
                        let mut itens = linha;
                        if cauda > 0 {
                            itens.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                            itens.truncate(cauda);
                        }
                        for (w, d) in itens { num += w * d; den += w; }
                    }
                    (lam as f64 * (num / den.max(1e-9)) * 2.0) as f32
                }).collect();
            } else {
                ctx.fiacao_custo.clear();
            }
        }
        if NEUROMOD_REGIOES.load(Ordering::Relaxed) {
            ctx.gates_w = genomes.iter().flat_map(|g| g.gates_w.iter().copied()).collect();
        }
        ctx.upload_engines(&engines);
        // Banco de τ: vai separado porque τ vive no genoma, não no `SovereignEngine`.
        // Uma vez por geração, como os pesos — τ é herdado e fixo na vida do ser.
        ctx.brain.upload_tau(&Genome::pack_taus(&genomes), n_seres);
        // Banco de sensores (FASE SENSORES, degrau 1): mesmo motivo do τ — vive no
        // genoma, sobe uma vez por geração para o pipeline ESPACIAL de percepção.
        // Com `SENSORS_EVOLVE_ON` no padrão (desligado), todo genoma tem
        // `SENSOR_DEFAULTS` e este upload sobe os hardcodes de hoje — equivalente a
        // nunca chamar o upload, só documenta a fonte única.
        ctx.percept.upload_sensors(&ctx.queue, &Genome::pack_sensors(&genomes), n_seres);
        // P1.7 (cérebro assíncrono) — metadados por neurônio (alfa/período/limiar/
        // ativação): mesmo motivo do τ/sensores, uma vez por geração.
        ctx.brain.upload_neuron_meta(&Genome::pack_neuron_meta(&genomes), n_seres);

        // ── P1.7, mecanismo 2 — quantos neurônios de L1 disparam por tick ──────
        //
        // Requisito da tarefa: reportar quantos neurônios EFETIVAMENTE recalculam
        // por tick (média e desvio), com a honestidade de que isto é REPRESENTAÇÃO,
        // não velocidade — ver `simulate_l1_fire_counts`. Simulado em CPU pura (sem
        // GPU, sem readback): o período de cada neurônio já está conhecido antes do
        // tick 0 rodar. Janela de 840 ticks = MMC(1..8), cobre um ciclo completo dos
        // períodos possíveis (1..8, clamp de `evolve()`) independente de onde a
        // amostragem começaria dentro da corrida de verdade.
        if !sem_periodo_neuronal {
            let periods_flat: Vec<f32> = genomes.iter()
                .flat_map(|g| g.neuron_period_l1.iter().copied())
                .collect();
            let janela = n_ticks.min(840).max(1);
            let (media_disp, desvio_disp) = universe::gpu::evolution::simulate_l1_fire_counts(
                &periods_flat, n_seres, janela);
            let total_neuronios = (n_seres * universe::gpu::evolution::N_L1_NEURONS) as f64;
            println!(
                "  [periodo] neurônios de L1 disparando/tick: {:.1} ± {:.1} de {:.0} ({:.1}%) \
                 · janela simulada {} ticks (REPRESENTAÇÃO, não velocidade — GPU roda todas as threads de qualquer forma)",
                media_disp, desvio_disp, total_neuronios,
                100.0 * media_disp / total_neuronios.max(1.0), janela);
        }

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
        // Mesma família: `last_intents` (decisão cacheada p/ ticks em que o cérebro
        // não dispara) e `brain_tick` atravessavam a fronteira de geração — os
        // primeiros ticks da gen nova reusavam a ÚLTIMA decisão da gen anterior
        // (achado da bateria de validação P1.9).
        ctx.last_intents.clear();
        ctx.brain_tick = 0;

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
                shift_cooldown: 0.0,
                shift_up_prev: 0.0,
                shift_down_prev: 0.0,
            nm_reserva: [1.0; 4],
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

            // ── Painel: snapshot de tick (E5, a cada 64 ticks) ────────────────────
            // Fica DEPOIS do laço por-ser de cima de propósito: states[i].life_left/fuel
            // já refletem este tick (vida ganha, combustível reabastecido no setor, etc.).
            if let Some(p) = &painel {
                if tick_i % 64 == 0 {
                    let snaps: Vec<painel::BeingSnapshot> = (0..n_seres).map(|i| {
                        let ph = &states[i].phys;
                        let vel = (ph.vel[0].powi(2) + ph.vel[2].powi(2)).sqrt();
                        painel::BeingSnapshot {
                            id: i,
                            x: ph.pos[0],
                            z: ph.pos[2],
                            vel,
                            life: (states[i].life_left.max(0.0) as f32 / life_cap() as f32).min(1.0),
                            vivo: states[i].life_left > 0.0,
                        }
                    }).collect();
                    let fuel_mediana = {
                        let mut fv: Vec<f64> = states.iter().take(n_seres).map(|s| s.fuel.max(0.0)).collect();
                        fv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        fv[fv.len() / 2] as f32
                    };
                    p.update_tick(tick_i as u64, snaps, fuel_mediana);
                }
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
            if elite_fitness_anterior_pista == Some(seed_gen) {
                let agora = fitness[0];
                let razao = if anterior > 1.0 { agora / anterior } else { f64::NAN };
                println!(
                    "  [elite] mesmo genoma: {:.0} → {:.0} (fitness, {:.0}% do anterior) · hodômetro={:.0}m",
                    anterior, agora, 100.0 * razao, dist_acum[0]
                );
                elite_reprod.push(razao);
            } else {
                println!(
                    "  [elite] pista mudou desde a última reavaliação (seed {} → {}) — razão \
                     não comparável, pulando",
                    elite_fitness_anterior_pista.unwrap_or(0), seed_gen
                );
            }
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
        let mut r = gen_stats(gen_abs, &prog_acum.iter().map(|p| p.max(0.0)).collect::<Vec<_>>(), lap_dist);
        let t_gen_s = t_gen.elapsed().as_secs_f64();

        hist_med.push(r.dist_med);
        hist_lap.push(lap_dist);
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
        // Validação cruzada 2026-08-15: o ≥1V/≥2V da linha oficial vinha de
        // prog_acum (projeção de arco, que PERDE metros em quinas de projeção) e
        // divergia do contador de setores em ordem (13 vs 22 ≥2V nas mesmas gens
        // de rpm-evolucao-s31). Setor em ordem é incheatável e é o mesmo critério
        // do currículo — a linha oficial agora reporta ELE. dist_med/max seguem
        // de prog_acum (metros contínuos são a métrica certa pra distância).
        r.n_ge1v = v1;
        r.n_ge2v = v2;
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

        // ── Painel: fim de geração (E5) ─────────────────────────────────────────
        // Fica AQUI (não logo após `hist_med.push`, mais acima) porque é onde `checkpoint`
        // e `n_portais` (para `setores`) já estão prontos — e `hist_med` já recebeu o push
        // desta geração, então `linear_slope` já enxerga o ponto atual.
        if let Some(p) = &painel {
            let setores_frac: Vec<f32> = (0..n_portais).map(|s| {
                checkpoint.iter().filter(|&&c| c > s).count() as f32 / n_seres.max(1) as f32
            }).collect();
            let slope = if hist_med.len() >= 2 { linear_slope(&hist_med).0 } else { 0.0 };
            p.update_gen(
                gen_abs as u32,
                r.dist_med,
                r.dist_max,
                r.mortality * 100.0,
                0.0, // obediência: não calculada neste laço — ver doc de `PainelState::obediencia_pct`.
                setores_frac,
                slope,
            );
        }

        // Velocidade mediana da população (m/s) — hoisted pra cima do bloco de relatório
        // de baixo (que a usa de novo) porque o gate do currículo (abaixo) também precisa
        // dela ANTES daquele bloco rodar. Mesma fórmula: distância acumulada / tempo vivo.
        let vmed: f64 = {
            let mut v: Vec<f64> = (0..n_seres).map(|i| {
                let seg = (states[i].ticks_alive.max(1) as f64) / 64.0;
                dist_acum[i] / seg
            }).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };

        // Circuito dominado? Então a próxima geração corre numa pista nova.
        //
        // P1.8, achado "currículo promove com métrica DIFERENTE da que o log mostra
        // como voltas reais": usa `v1` (voltas_reais ORDENADO, setor a setor) em vez
        // de `r.n_ge1v` (hodômetro/prog_acum ≥ lap_dist, que faz aliasing de posição
        // — `get_track_progress` busca o segmento mais PRÓXIMO, sem exigir ordem de
        // visita) — os dois calculam a mesma pergunta por caminhos genuinamente
        // diferentes, e só `v1` bate com o texto que o log já imprime como "voltas
        // reais" (bloco `[setores]`, logo acima).
        //
        // Achado "UM ser rastejando promove a turma inteira" (`--trocar-ao-completar 1`):
        // a promoção agora exige UMA das duas rotas, não só o `N` cru do usuário:
        //   (a) `v1` cobre pelo menos 25% da população (amplitude — não foi 1 sortudo), OU
        //   (b) `v1` já bate o `N` configurado E a mediana de velocidade da geração supera
        //       `curriculo_veloc_min` (ritmo — quem completou não estava rastejando).
        // Isso reforça, nunca afrouxa, o `N` que o usuário pediu: o piso populacional só
        // AUMENTA o `N` efetivo quando 25% da população é maior que ele.
        let piso_populacional = (((n_seres as f64) * 0.25).ceil() as usize).max(1);
        let limiar_efetivo = trocar_ao_completar.max(piso_populacional);
        let cond_populacao = v1 >= limiar_efetivo;
        let cond_veloc = v1 >= trocar_ao_completar && vmed >= curriculo_veloc_min;
        if trocar_ao_completar > 0 && (cond_populacao || cond_veloc) {
            // Currículo de TETO DE MARCHA: a promoção sobe o corpo ANTES de rodar a
            // pista — um degrau por vez. Só quando o teto está no máximo a promoção
            // volta a rodar a pista.
            let teto_atual = CAMBIO_TETO.load(Ordering::Relaxed);
            let teto_max   = CAMBIO_TETO_MAX.load(Ordering::Relaxed);
            if CAMBIO_CURRICULO.load(Ordering::Relaxed)
                && CAMBIO_MODO.load(Ordering::Relaxed) > 0
                && teto_atual < teto_max
            {
                CAMBIO_TETO.store(teto_atual + 1, Ordering::Relaxed);
                println!("  [cambio] população completou a volta ({v1} seres) — teto de marcha {} → {} (pista continua {})",
                         teto_atual, teto_atual + 1, world_seed_atual);
            } else {
            idx_curriculo += 1;
            let proxima = pista_da_gen(&pistas, idx_curriculo);
            let via = if cond_populacao {
                format!("população (v1={} ≥ {} = max(N={}, 25%={}))",
                        v1, limiar_efetivo, trocar_ao_completar, piso_populacional)
            } else {
                format!("velocidade (v1={} ≥ N={} · mediana {:.2}m/s ≥ piso {:.2}m/s)",
                        v1, trocar_ao_completar, vmed, curriculo_veloc_min)
            };
            println!(
                "  [curriculo] {} seres completaram a volta via {} — pista {} → {}",
                v1, via, world_seed_atual, proxima);
            }
        } else if trocar_ao_completar > 0 && v1 > 0 {
            // Visível mesmo quando NÃO promove: sem isto, "por que não trocou" exige
            // reler o código toda vez que alguém completa a volta e a pista não muda.
            println!(
                "  [curriculo] {} seres completaram a volta, mas nem população (precisa {}) \
                 nem velocidade (mediana {:.2}m/s, precisa {:.2}m/s com N={}) bateram o piso \
                 — pista continua {}",
                v1, limiar_efetivo, vmed, curriculo_veloc_min, trocar_ao_completar, world_seed_atual);
        }

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
        // (`vmed` já foi calculada acima, antes do gate do currículo — reusa aqui.)
        {
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
        if NEUROMOD_REGIOES.load(Ordering::Relaxed) {
            println!("  [neuromod] gates ser0 [visão,parede,corpo,assoc] = [{:.2},{:.2},{:.2},{:.2}] · reserva [{:.2},{:.2},{:.2},{:.2}]",
                ctx.ultimo_gate[0], ctx.ultimo_gate[1], ctx.ultimo_gate[2], ctx.ultimo_gate[3],
                states[0].nm_reserva[0], states[0].nm_reserva[1], states[0].nm_reserva[2], states[0].nm_reserva[3]);
        }
        println!("  [veloc] mediana {:.2} m/s ({:.1} km/h) | pico {:.1} m/s | marcha máx {} | vida média {:.0}% da geração",
                     vmed, vmed * 3.6,
                     VEL_PICO.load(Ordering::Relaxed), MARCHA_PICO.load(Ordering::Relaxed),
                     100.0 * vida_med / n_ticks as f64);
        VEL_PICO.store(0.0, Ordering::Relaxed);
        MARCHA_PICO.store(0, Ordering::Relaxed);
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
            // Contexto físico alinhado 1:1 a `l1`/`l2` (`--regioes`) — ver doc de
            // `regiao_l1`/`regiao_l2` em context.rs. Sempre tomado aqui (mesmo sem
            // `--regioes`) para não vazar amostra entre gerações, mesmo raciocínio
            // do `mem::take` das linhas acima.
            let regiao_l1 = std::mem::take(&mut ctx.regiao_l1);
            let regiao_l2 = std::mem::take(&mut ctx.regiao_l2);

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

            // ── ESPARSIDADE CONTEXTUAL (E1-E4 do protocolo) ────────────────────
            //
            // Pergunta diferente da de cima: `[ativacao]` mede a SAÚDE da camada (morto,
            // saturado, ganho); isto mede QUEM dispara para QUAL situação. A distinção
            // importa porque a esparsidade ESTÁTICA (quais neurônios são globalmente
            // importantes) esta refutada — com os 8 "mais importantes" de 128 o
            // desempenho fica pior que o baseline, e a nossa atencao por canal ja media
            // uniforme (0,52-0,66%). O que existe e a esparsidade CONTEXTUAL.
            //
            // Limiar sempre impresso: `tanh` nao zera exatamente, entao o numero depende
            // da escolha, e escolha que muda resultado tem de aparecer no relatorio.
            {
                use universe::analysis::contextual as ctx;
                const LIMIAR: f32 = 0.05;
                // Entradas da Lei do Garimpo. `ganho` = quanto o melhor da geracao
                // superou o melhor da anterior; `disp` = desvio da fitness na populacao,
                // que e a escala do ruido contra a qual o ganho e comparado.
                let ganho_fitness_geracao: Option<f32> = elite_fitness_anterior
                    .map(|ant| (fitness[0] - ant).max(0.0) as f32);
                let disp_fitness: f32 = {
                    let vals: Vec<f32> = fitness.iter().map(|f| *f as f32).collect();
                    universe::math::stats::std_dev(&vals)
                };
                for (nome, amostras) in [("L1", &l1), ("L2", &l2)] {
                    if amostras.len() < 2 { continue; }
                    let e = ctx::esparsidade(nome, amostras, LIMIAR);
                    let p = ctx::persistencia(nome, amostras, LIMIAR, 37);
                    let q = ctx::quente_frio(nome, amostras, LIMIAR);
                    println!(
                        "  [contextual:{}] limiar={:.2} · ativos {:.1}%±{:.1} · saturado {:.1}%",
                        nome, LIMIAR, e.fracao_ativa_media * 100.0,
                        e.fracao_ativa_sigma * 100.0, e.saturacao_media * 100.0);
                    println!(
                        "  [contextual:{}] persistencia {:.1}% · baseline {:.1}% · ganho {:+.1} pts · churn {:.1}%",
                        nome, p.persistencia * 100.0, p.baseline_aleatorio * 100.0,
                        p.ganho_temporal * 100.0, p.churn * 100.0);
                    println!(
                        "  [contextual:{}] {} neuronios cobrem 80% das ativacoes · {} nunca · {} sempre",
                        nome, q.neuronios_para_80pct, q.nunca_ativos, q.sempre_ativos);
                    // Armadilha nº 1 do protocolo: `tanh` nao zera, entao "ativo" e uma
                    // escolha de limiar. A curva mostra se a conclusao sobrevive a ela.
                    let sens = ctx::sensibilidade_ao_limiar(amostras, &[0.01, 0.05, 0.2, 0.5, 0.9]);
                    let curva: Vec<String> = sens.iter()
                        .map(|(l, f)| format!("{l:.2}→{:.0}%", f * 100.0)).collect();
                    println!("  [contextual:{}] sensibilidade ao limiar: {}", nome, curva.join(" · "));

                    // E3 — co-ativacao. E a base EMPIRICA para dividir o cerebro em
                    // regioes: os grupos tem de sair de neuronios que disparam juntos,
                    // nao de caixas desenhadas a priori. Excesso ~0 significa que cada
                    // neuronio decide sozinho — sem estrutura de grupo, nao ha regiao a
                    // separar (foi o que pesos aleatorios deram no protocolo: -0,1%).
                    let co = ctx::coativacao(nome, amostras, LIMIAR, 5);
                    println!(
                        "  [contextual:{}] co-ativacao {:.3} vs independente {:.3} · excesso {:+.3}",
                        nome, co.coativacao_media, co.esperado_independente, co.excesso);
                    if !co.pares_fortes.is_empty() {
                        let ps: Vec<String> = co.pares_fortes.iter()
                            .map(|(i, j, e)| format!("({i},{j}){e:+.2}")).collect();
                        println!("  [contextual:{}] pares mais acoplados: {}", nome, ps.join(" "));
                    }

                    // CAPACIDADE — quanto espaco este cerebro realmente USA.
                    //
                    // Leonardo: "podemos avaliar o quanto de espaco eles realmente
                    // precisam, e se o modelo ficar com pouco espaco a gente vai
                    // aumentando". `participation_ratio` responde isso a partir das
                    // ativacoes reais: PR = (Σλ)²/Σλ² sobre o espectro da covariancia.
                    // PR perto de N = usa todas as dimensoes (saturado, pede crescer);
                    // PR << N = ha capacidade ociosa, crescer nao ajudaria.
                    let n_neur = amostras[0].len();
                    if let Some(pr) = universe::math::linalg::participation_ratio(amostras) {
                        println!(
                            "  [contextual:{}] capacidade: PR {:.1}/{} ({:.0}% das dimensoes em uso)",
                            nome, pr, n_neur, 100.0 * pr / n_neur as f32);

                        // LEI DO GARIMPO — crescer o cerebro se justifica AGORA?
                        //
                        // Leonardo: "o modelo vai chegar em um maximo e vamos precisar
                        // dessa evolucao". O veredito precisa de DUAS condicoes, e por
                        // isso a funcao existia mas nunca foi chamada — faltava plugar:
                        //
                        //  (A) capacidade SATURADA (PR encostando na largura): se ha
                        //      dimensao ociosa, crescer so adiciona peso morto;
                        //  (B) o ganho de fitness exceder o limiar de ruido de busca,
                        //      sigma*sqrt(2*ln M) com M = numero de alternativas testadas
                        //      (a populacao). Sem isso, "melhorou" e o melhor de M sorteios,
                        //      nao competencia — o mesmo erro que o Deflated Sharpe corrige
                        //      em financas.
                        //
                        // Um veredito por camada, porque cada uma satura no seu tempo.
                        let ganho = ganho_fitness_geracao.unwrap_or(0.0);
                        let v = universe::brain_capacity::growth_verdict(
                            amostras, n_neur, 0.80, ganho, disp_fitness, n_seres);
                        let (rotulo, acao) = match v {
                            universe::brain_capacity::Verdict::Grow =>
                                ("CRESCER", "capacidade saturada e ganho acima do ruido de busca"),
                            universe::brain_capacity::Verdict::WastedSpace =>
                                ("nao crescer", "ha dimensoes ociosas — crescer desperdicaria"),
                            universe::brain_capacity::Verdict::GainNotReal =>
                                ("nao crescer", "saturada, mas o ganho nao passa do limiar do Garimpo"),
                            universe::brain_capacity::Verdict::InsufficientData =>
                                ("indefinido", "amostras insuficientes para estimar a covariancia"),
                        };
                        println!("  [garimpo:{}] {} — {}", nome, rotulo, acao);
                    }
                }
            }

            // ── REGIÕES DA PISTA × ATIVAÇÃO (`--regioes`, src/analysis/regioes.rs) ──
            //
            // Leonardo: "o cérebro vai trabalhar em dimensões, neurônios e partes
            // diferentes conforme ele está na pista, e isso precisa ser medido."
            // Nunca cruzamos ativação com POSIÇÃO — isto cruza as MESMAS amostras
            // `l1`/`l2` de cima com o contexto físico que `ctx.tick()` grava em
            // lockstep (`regiao_l1`/`regiao_l2`): curvatura local, velocidade,
            // distância à parede.
            if regioes_on {
                use universe::analysis::regioes;
                // Limiares SEMPRE reportados (armadilha nº1 do protocolo de
                // `contextual.rs`: nada aqui zera exatamente, o número depende da
                // escolha). `LIMIAR_ATIVACAO_COESAO` reusa o mesmo valor do bloco
                // `[contextual]` acima para a coesão ser comparável ao E3 geral.
                const LIMIAR_CURVA: f32 = 0.02; // 1/m — raio < 50 m conta como "curva"
                const LIMIAR_ESPECIALIZACAO: f32 = 0.30;
                const LIMIAR_ATIVACAO_COESAO: f32 = 0.05;

                for (nome, amostras, contexto) in [
                    ("L1", &l1, &regiao_l1),
                    ("L2", &l2, &regiao_l2),
                ] {
                    match regioes::analisar(nome, amostras, contexto,
                        LIMIAR_CURVA, LIMIAR_ESPECIALIZACAO, LIMIAR_ATIVACAO_COESAO)
                    {
                        Some(r) => {
                            println!(
                                "  [regioes:{}] limiar_curva={:.3} 1/m · amostras reta={} curva={}",
                                nome, r.limiar_curva, r.n_reta, r.n_curva);
                            println!(
                                "  [regioes:{}] especializacao >= {:.2}: {}/{} neuronios · idx {:?}",
                                nome, r.limiar_especializacao, r.indices_especializados.len(),
                                r.por_neuronio.len(), r.indices_especializados);
                            let mut ord = r.por_neuronio.clone();
                            ord.sort_by(|a, b| b.indice_especializacao
                                .partial_cmp(&a.indice_especializacao).unwrap());
                            let top: Vec<String> = ord.iter().take(5).map(|e| format!(
                                "#{}(idx={:.2} corr_curva={:+.2} corr_vel={:+.2} corr_parede={:+.2})",
                                e.indice, e.indice_especializacao, e.correlacao_curvatura,
                                e.correlacao_velocidade, e.correlacao_dist_parede)).collect();
                            println!("  [regioes:{}] top especializados: {}", nome, top.join(" "));
                            match &r.coesao {
                                Some(co) => println!(
                                    "  [regioes:{}] coesao dos especializados: co-ativacao {:.3} vs independente {:.3} · excesso {:+.3}",
                                    nome, co.coativacao_media, co.esperado_independente, co.excesso),
                                None => println!(
                                    "  [regioes:{}] coesao: N/A (menos de 2 especializados ou menos de 2 amostras de curva)",
                                    nome),
                            }
                        }
                        None => println!(
                            "  [regioes:{}] sem dados (amostras={} contexto={} — desalinhado ou vazio)",
                            nome, amostras.len(), contexto.len()),
                    }
                }
            }

            // Acumula para o painel (E5, `--painel`) enquanto imprime — mesmos números,
            // um destino a mais. `painel_layers`/`painel_alarms` só são consumidos se
            // `painel` for `Some`; construir sempre é barato (poucas dezenas de floats).
            let mut painel_layers: Vec<painel::LayerSnapshot> = Vec::with_capacity(2);
            let mut painel_alarms: Vec<String> = Vec::new();

            // Ultimo instante de cada camada: `l1`/`l2` sao [amostra][neuronio] achatados,
            // entao a ultima fatia de tamanho N e o estado no fim da geracao. E o "agora"
            // que o painel anima; os agregados ao lado contam a historia.
            let mut ultima_amostra_camada: std::collections::HashMap<String, Vec<f32>> =
                std::collections::HashMap::new();
            // `l1`/`l2` sao Vec<Vec<f32>>: uma linha por amostra, uma coluna por neuronio.
            for (nome, buf) in [("L1", &l1), ("L2", &l2)] {
                if let Some(ultima) = buf.last() {
                    ultima_amostra_camada.insert(nome.to_string(), ultima.clone());
                }
            }

            for (stats, alarms) in [(&l1_stats, &l1_alarms), (&l2_stats, &l2_alarms)] {
                universe::analysis::activation::print_stats_report(stats, alarms);
                let n_neurons = stats.neurons.len().max(1);
                let dead_pct: f32 = stats.neurons.iter().map(|s| s.dead_pct).sum::<f32>()
                    / n_neurons as f32;
                let saturad_pct: f32 = stats.neurons.iter().map(|s| s.saturated_pct).sum::<f32>()
                    / n_neurons as f32;
                let snr_pct: f32 = stats.neurons.iter().map(|s| s.snr_i8).sum::<f32>()
                    / n_neurons as f32;
                println!(
                    "  [ativacao:{}] amostras={} dead_pct={:.1}% saturad_pct={:.1}% \
                     raio_espectral={} rank_efetivo={} alarmes={}",
                    stats.layer_name, stats.n_samples, dead_pct, saturad_pct,
                    stats.spectral_radius.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into()),
                    stats.effective_rank.map(|k| format!("{k}/{n_neurons}"))
                         .unwrap_or_else(|| "n/a (pesos nao entregues)".into()),
                    alarms.len(),
                );
                if painel.is_some() {
                    painel_layers.push(painel::LayerSnapshot {
                        name: stats.layer_name.clone(),
                        dead_pct,
                        saturated_pct: saturad_pct,
                        // Painel usa 0/0 como sentinela de "nao medido"; o rotulo textual
                        // fica no relatorio de texto, que e onde a leitura humana acontece.
                        spectral_radius: stats.spectral_radius.unwrap_or(0.0),
                        effective_rank: stats.effective_rank.unwrap_or(0),
                        snr_i8: snr_pct,
                        histogram: stats.histogram.clone(),
                        // Ativacao viva: `live` e o ultimo valor amostrado de cada
                        // neuronio (o "agora" da camada); `mean_per_neuron` e a media da
                        // geracao (o "sempre"). Ver os dois lado a lado distingue neuronio
                        // que RESPONDE a pista de neuronio que so tem vies alto.
                        live: ultima_amostra_camada.get(&stats.layer_name)
                                 .cloned().unwrap_or_default(),
                        mean_per_neuron: stats.neurons.iter().map(|n| n.mean).collect(),
                    });
                    for a in alarms {
                        painel_alarms.push(format!(
                            "{}: {:?} valor={:.2} limiar={:.2}", a.layer, a.kind, a.value, a.threshold
                        ));
                    }
                }
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
                if painel.is_some() {
                    painel_alarms.push(format!(
                        "NaN/Inf detectado no estado L1 de {} ser(es) nesta geração", nan_seres.len()
                    ));
                }
            }
            if let Some(p) = &painel {
                p.update_activation(painel_layers, painel_alarms);
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

        // ── Held-out (P1) ───────────────────────────────────────────────────────
        //
        // Roda DEPOIS de todo relatório desta geração que consome
        // `ctx.l1_samples`/`l2_samples`/`slow_samples`/`policy_samples`/`nan_seres`
        // (blocos `[lento]` e `[ativacao:*]` acima, que já fizeram `mem::take`) —
        // colocar antes contaminaria o relatório de ativação DESTA geração com
        // ticks da pista reservada. E roda ANTES de `evolve()` logo abaixo — item 4
        // da TAREFA: a seleção nunca vê este número (ver doc de `avaliar_held_out`
        // para o porquê).
        if let Some(ho_seed) = pista_held_out {
            if gen_abs % held_out_cada == 0 {
                // P1.8, achado "avaliar_held_out contamina AUDIT_ACUM": `ctx.tick()`
                // acumula em AUDIT_ACUM (context.rs, gate `AUDITAR`) sempre que a flag
                // está ligada — sem distinguir se o tick veio do laço de treino ou
                // daqui. Desliga a flag SÓ durante os ticks do held-out (pista
                // diferente da de treino) e restaura logo depois — mesmo padrão que
                // `avaliar_held_out` já usa pra `l1_samples`/`l2_samples`/etc (limpos
                // antes, main.rs ~linha 266-280), mas resolvido pelo toggle que já
                // existe (`AUDITAR`, statics.rs) em vez de clonar o `AuditAcum` inteiro.
                // Bônus: held-out para de pagar o readback de visão que nunca usa.
                AUDITAR.store(false, Ordering::Relaxed);
                let held_med = avaliar_held_out(
                    &mut ctx, n_seres, n_ticks, track_width, same_start,
                    &modo_fitness, ho_seed, &kan_knots_global,
                );
                AUDITAR.store(auditar_visao, Ordering::Relaxed);
                // Mesma classe do toggle AUDITAR acima: o held-out roda ctx.tick()
                // com os MESMOS statics de telemetria — sem este reset, o pico de
                // velocidade/marcha da pista RESERVADA vazava pro [veloc] da
                // geração de treino seguinte (achado da bateria de validação P1.9).
                VEL_PICO.store(0.0, Ordering::Relaxed);
                MARCHA_PICO.store(0, Ordering::Relaxed);
                // `avaliar_held_out` trocou a pista na GPU para `ho_seed` — a
                // próxima geração detecta isso (`seed_gen != world_seed_atual`,
                // topo do laço) e recarrega a pista de treino sozinha, sem
                // precisar de nenhum código de "restaurar" aqui.
                world_seed_atual = ho_seed;

                // MEDIANA dos vivos (item 3 da TAREFA) — mesmo critério "vivo"
                // (`life_left > 0.0`) usado no resto do relatório desta geração
                // (ex.: bloco `sem_vida` logo abaixo).
                let treino_vivos: Vec<f64> = (0..n_seres)
                    .filter(|&i| states[i].life_left > 0.0)
                    .map(|i| fitness[i])
                    .collect();
                let treino_med = if treino_vivos.is_empty() {
                    // Ninguém vivo no treino: cai para todos, mesmo critério de
                    // `avaliar_held_out` — mantém as duas médias no mesmo padrão.
                    let mut f = fitness.clone();
                    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    f[f.len() / 2]
                } else {
                    let mut v = treino_vivos;
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[v.len() / 2]
                };
                let razao = if treino_med > 1.0 { held_med / treino_med } else { f64::NAN };
                println!(
                    "  [generalizacao] gen={:4} treino={:.0}m held={:.0}m razao={:.3}  \
                     (pista treino=seed {} · held-out=seed {})",
                    gen_abs, treino_med, held_med, razao, seed_gen, ho_seed,
                );
            }
        }

        // O genoma que vai para o slot 0 da próxima geração é o melhor DESTA.
        elite_fitness_anterior = Some(
            fitness.iter().cloned().fold(f64::MIN, f64::max)
        );
        elite_fitness_anterior_pista = Some(seed_gen);

        hist_results.push(r);

        // ── Evolução (seleção + mutação) ───────────────────────────────────────
        if generation_idx + 1 < n_gens {
            // SELEÇÃO sobre a fitness escolhida, não sobre o hodômetro.
            //
            // Aqui estava `&dist_acum` fixo: eu tinha trocado só o que o relatório EXIBE
            // e deixado a seleção premiando hodômetro. As duas condições `--fitness dist`
            // e `--fitness prog` saíam idênticas ao dígito — e eu quase registrei isso
            // como "a fitness não importa". Ela nunca tinha mudado.
            genomes = evolve(&genomes, &fitness, &evo_params, gen_abs, pop_seed);
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
        let ld = hist_lap.get(i).copied().unwrap_or(lap_dist);
        println!("  gen {:4}: med={:7.0}m ({:.3}V) max={:7.0}m ({:.3}V)",
            gen_offset + i, med, med / ld, max, max / ld);
    }

    println!();

    // Imprime relatório da auditoria se ela foi ativada.
    if AUDITAR.load(Ordering::Relaxed) || audit::n_ticks_acumulados() > 0 {
        audit::auditoria_relatorio();
    }

    println!("=== FIM gpu_evolve ===");
}

