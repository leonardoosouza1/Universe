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
//!       --beings 60 --ticks 20000 --gens 100
//!
//!   # Para retomar de onde parou:
//!   env -u DISPLAY nice -n 19 ./target/release/gpu_evolve \
//!       --beings 60 --ticks 20000 --gens 100 --resume

use std::sync::atomic::Ordering;
use std::time::Instant;

mod audit;
mod body_evolution;
mod brain_backend;
mod cli_flags;
#[cfg(test)]
mod cli_flags_tests;
mod context;
mod driving_diagnostics;
mod ecology_runtime;
mod escada;
mod fantasma;
mod genome_diagnostics;
mod genome_io;
mod laps;
#[cfg(test)]
mod laps_tests;
mod native_action;
mod objective_cost;
mod organism_checkpoint;
mod painel;
mod piloto_forja;
mod piloto_linear;
mod pilots;
mod pistas;
mod reverse_controller;
mod school_batch;
mod starting_grid;
mod statics;
mod survival;
#[cfg(test)]
mod tests;
mod types;
mod visual_telemetry;
mod wall_probes;

use context::GpuCtx;
use genome_io::{load_genome, population_seed_for, save_genome};
use pistas::{parse_pistas, pista_da_gen, spec_da_gen};
use starting_grid::{canonical_start, formation_slots};
use statics::*;
use types::BeingState;

use universe::engine::SovereignEngine;
use universe::evolution::dna_genesis::default_kan_knots;
use universe::gpu::evolution::{EvoParams, GenResult, Genome, evolve, gen_stats, linear_slope};
use universe::gpu::taxi::TaxiMode;
use universe::world::VoxelWorld;

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
use universe::gpu::physics::pos_of;

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
/// - `--curve-portal`: held-out sempre usa setor UNIFORME. Comparar `razao` exige
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
/// Direção de largada em um waypoint, à prova do FECHAMENTO DO LAÇO.
///
/// A lista de waypoints repete o primeiro ponto no fim para fechar o circuito
/// (`world.rs`, `waypoints.push(waypoints[0])`). Nascer nesse último índice dava
/// `wp == wp_next` → `dx = dz = 0` → com o antigo `.max(1e-6)` o versor saía
/// `[0,0,0]`: orientação DEGENERADA. Efeito medido em 2026-08-16 (pista 8200,
/// `--random-start` sorteou wi=105/106): velocidade EXATAMENTE 0,00 a
/// corrida inteira com acelerador 0,4-0,6 aplicado e RPM girando, hodômetro 0 m,
/// 48/48 mortos e `fitness 2377..2377 · valores distintos 1/48` — uma geração
/// inteira de SELEÇÃO CEGA. Reproduzido em 10 s com `--seed 405` (wi=105) contra
/// `--seed 285` (wi=50, normal: 19/48 notas distintas, mortalidade 0%).
///
/// Conserto: procurar adiante (com wrap) o primeiro waypoint REALMENTE distinto.
/// Nunca devolve versor nulo em pista válida; se a pista inteira for um ponto só,
/// devolve [1,0,0] e avisa — melhor um ser andando em linha reta que 48 pinos.
/// Fuel capacity is sized from the actual refill interval.
///
/// Dois bugs foram pagos aqui em 2026-08-16, e os dois eram invisíveis no log:
/// 1. a conta usava a VOLTA (`L/18,75 × 0,003 × k`) enquanto o reabastecimento
///    acontece a cada setor (24 por volta) — tanque ~12× maior que o intervalo em
///    que é recarregado, logo NUNCA prendia;
/// 2. a conta só rodava na TROCA de pista, então em corrida de pista única (e na
///    primeira geração de qualquer bateria) `--fuel-auto k` não fazia nada: k=2,0,
///    1,0 e 0,7 davam resultado IDÊNTICO (1923 m, mortalidade 0%) — o canário de
///    números iguais pegou.
/// Juntos explicam por que a bateria-escada teve 34 de 35 gerações com mortalidade
/// 0% mesmo com combustível, ruído, dano e clima ligados.
///
/// `k = 1` is the fuel needed to cover that interval at 18.75 m/s under full
/// throttle consumption (0.006 units/s). Sector refill uses L/24; lap-only refill
/// uses L. The same function therefore cannot silently grant a lap-sized tank to a
/// sector-refill run or a sector-sized tank to a lap-refill run.
fn automatic_fuel_tank(lap_dist: f64, k: f32, refill_each_sector: bool) -> f32 {
    let refill_distance_m = if refill_each_sector {
        lap_dist as f32 / N_CHECKPOINTS as f32
    } else {
        lap_dist as f32
    };
    (refill_distance_m / 18.75) * 0.006 * k
}

/// Base tyre-thermal parameters for a dose, DC-normalised — the population's shared
/// shape, before the per-being spread.
///
/// The dose scales the DEPTH of the grip curve: `floor = 1 - gain * (1 - 0.70)`, so
/// gain 0 gives floor 1.0 (`grip_factor` is identically 1: the mechanism is exactly
/// off, not approximately off) and gain 1 gives the legacy floor 0.70.
///
/// `max_factor` is DERIVED, not chosen. `normalize_dc` finds the trim that makes mean
/// applied grip exactly 1.0 over the reference band, and that trim grows as the floor
/// drops (deeper bell, lower raw mean). A fixed cap of 1.20 is enough for gain 1 but
/// BINDS at gain 2 — the trim would be clamped, the mean would land below 1.0, and the
/// "ON" arm would silently be "less grip everywhere". That is the DC-gain confound this
/// project already paid for, reappearing one dose later. So the cap is set just above
/// the trim the band requires (5% slack) and can never bind inside the band.
///
/// Concretely: raw mean over the band is `g0`; the trim converges to `1/g0`; the peak
/// applied factor is `1 * (1/g0)`, which is `< 1.05/g0 = max_factor`.
fn thermal_base_params(gain: f32) -> universe::tire_thermal::ThermalParams {
    use universe::tire_thermal as th;
    let floor = (1.0 - gain * (1.0 - th::DEFAULT_FLOOR)).clamp(0.0, 1.0);
    let band = th::reference_band();
    let mut p = th::ThermalParams {
        floor,
        max_factor: f32::INFINITY,
        ..Default::default()
    };
    let g0 = th::dc_gain(&band, &p).max(1e-6);
    p.max_factor = 1.05 / g0;
    th::normalize_dc(&p, &band)
}

fn frente_do_waypoint(wps: &[bevy::math::Vec2], wi: usize) -> [f32; 3] {
    let n = wps.len();
    let wp = wps[wi];
    for passo in 1..n {
        let cand = wps[(wi + passo) % n];
        let dx = cand.x - wp.x;
        let dz = cand.y - wp.y;
        let len = (dx * dx + dz * dz).sqrt();
        if len > 1e-3 {
            return [dx / len, 0.0, dz / len];
        }
    }
    eprintln!("[AVISO] largada degenerada: todos os {n} waypoints coincidem com o índice {wi}");
    [1.0, 0.0, 0.0]
}

/// Retorna a MEDIANA da fitness held-out entre os seres que terminaram VIVOS
/// (`life_left > 0`) — item 3 da TAREFA ("razão usa mediana dos vivos"). Se
/// ninguém sobreviver, cai para a mediana de TODOS (avisado via `eprintln!`) em
/// vez de indexar um vetor vazio.
fn avaliar_held_out(
    ctx: &mut GpuCtx,
    genomes: &[Genome],
    evolving_body: bool,
    n_seres: usize,
    n_ticks: usize,
    track_width: f32,
    same_start: bool,
    modo_fitness: &str,
    wall_cost: f32,
    impact_cost: f32,
    held_out_seed: u64,
    kan_knots: &[[f32; universe::kan::KAN_OUT_KNOTS]; 3],
) -> f64 {
    // Held-out ticks must never enter the training replay observation stream.
    let saved_diagnostics = ctx.driving_diagnostics.take();
    let organism_taxi = ctx.organism.is_some();
    let organism_saved = (ctx.organism.take(), ctx.taxi.take(), ctx.organism_feedback.take(), std::mem::take(&mut ctx.organism_samples), ctx.traffic.take(), std::mem::take(&mut ctx.traffic_samples));
    let mut world_ho = VoxelWorld::new(held_out_seed, track_width);
    let held_city = organism_taxi.then(|| universe::organism_city::build_city(&mut world_ho));
    let held_stations = if organism_taxi {
        let stations = held_city.as_ref().unwrap().stations.clone();
        universe::organism_scenario::validate_city(&world_ho, &stations).expect("held-out taxi topology");
        stations
    } else { Vec::new() };
    // Troca a geometria na GPU (grade/fluxo/alturas/heightmap) sem recriar device
    // nem pipeline — ver `GpuCtx::reload_world`. Reseta o estado recorrente (CfC)
    // para que os seres cheguem na pista reservada sem memória da pista de treino
    // desta geração — mesmo contrato de "reset entre gerações" que já existe no
    // laço principal (main(), bloco "Reseta CfC entre gerações").
    ctx.reload_world(&world_ho);
    ctx.reset_cfc(n_seres);
    for i in 0..n_seres {
        ctx.brain.reset_cfc(i);
        if organism_taxi { ctx.brain.reset_lifetime(i); }
    }
    // ── HOSTILE WORLD: suspended for the held-out run (BODY PHASE) ────────────
    //
    // Same contamination the `l1_samples.clear()` below already fixes, one layer
    // deeper: `ctx` is shared with the training loop, so without this the D7 stations
    // would still be sitting at the coordinates of the TRAINING track — placed from
    // `world.waypoints_world` a moment ago, now floating over a different geometry —
    // and the predator would be hunting on the reserved track too. Both would move the
    // held-out number for reasons that have nothing to do with the route generalising,
    // which is the only question this function asks.
    //
    // It is also the choice already recorded for `--fuel` in this function's header:
    // held-out measures whether the ROUTE generalises, not whether tank management
    // does. Scarce stations, a predator and a tyre thermal are three more survival
    // budgets, and they belong on the same side of that line.
    //
    // Saved and restored rather than rebuilt: the station charges are mid-generation
    // state, and rebuilding them would silently refill the field the training
    // generation had already drained.
    let hostile_saved = (
        std::mem::take(&mut ctx.stations),
        ctx.predator.take(),
        std::mem::take(&mut ctx.thermal),
        std::mem::take(&mut ctx.sensor_cost),
    );
    // C1 — the hippocampus is suspended for the held-out run, on the same side of the
    // same line as fuel, the stations and the predator above.
    //
    // Suspended and not merely paused: this loop never calls `accumulate` or `cross`, so
    // leaving the bank in place would have every being reading, tick after tick, the
    // memory of the TRAINING track's sectors while driving a different geometry — sector
    // 7 of one track is not sector 7 of another. Held-out asks whether the ROUTE
    // generalises; a frozen memory of the wrong track moves that number for a reason that
    // has nothing to do with the question. The writer goes with it so the generation's
    // `hip_gen<N>.f32` does not gain rows from a track it does not describe.
    //
    // Taken and restored, not rebuilt: the 24 slots are mid-generation state that the
    // training loop is still accumulating into.
    let hip_saved = (
        std::mem::take(&mut ctx.hip_mem),
        std::mem::take(&mut ctx.hip_genes),
        ctx.hip_writer.take(),
    );
    let telemetry_saved = (
        ctx.tel_dry_beings,
        ctx.tel_predator_ticks,
        ctx.tel_predator_life,
        ctx.tel_station_fuel,
        ctx.tel_tire_temp_sum,
        ctx.tel_tire_temp_n,
    );

    let lap_dist = world_ho.lap_distance as f64;
    let n_wp = world_ho.waypoints_world.len();
    let held_out_start = if same_start && !organism_taxi {
        Some(canonical_start(&world_ho).unwrap_or_else(|error| {
            panic!("invalid held-out canonical start: {error}");
        }))
    } else {
        None
    };
    // Held-out evaluation keeps the slot map fixed (rotation 0): it compares genomes
    // on one identical layout, and the layout itself is recorded by the manifest.
    let city_spawns = organism_taxi.then(|| universe::organism_city::spawn_slots(&world_ho, &held_stations, n_seres).expect("held-out city spawn capacity"));
    let mut states: Vec<BeingState> = (0..n_seres)
        .map(|i| {
            let (pos, fwd) = if let Some(spawns) = &city_spawns {
                (spawns[i].position, spawns[i].forward)
            } else if let Some(slot) = held_out_start {
                (
                    [slot.position.x, 0.0, slot.position.y],
                    [slot.forward.x, 0.0, slot.forward.y],
                )
            } else {
                let wi = (i * n_wp / n_seres).min(n_wp - 1);
                let wp = world_ho.waypoints_world[wi];
                (
                    [wp.x, 0.0, wp.y],
                    frente_do_waypoint(&world_ho.waypoints_world, wi),
                )
            };
            BeingState {
                phys: body_evolution::at_birth(&genomes[i], evolving_body, pos, fwd),
                prev_steer: 0.0,
                stagnation_ticks: 0,
                wall_ticks: 0,
                collision_energy_j: 0.0,
                impact_count: 0,
                hole_hits: 0,
                hole_cooldown_ticks: 0,
                center_prog: 0.0,
                sensor_steer: 0.0,
                sensor_thr: 0.0,
                sensor_brk: 0.0,
                wp_near: 0,
                fuel: body_evolution::tank_capacity(
                    &genomes[i],
                    evolving_body,
                    FUEL_TANK.load(Ordering::Relaxed),
                ),
                fuel_capacity: body_evolution::tank_capacity(
                    &genomes[i],
                    evolving_body,
                    FUEL_TANK.load(Ordering::Relaxed),
                ),
                ticks_alive: 0,
                // Vida CHEIA (ver doc da função: held-out mede rota, não orçamento de fôlego).
                life_left: n_ticks as f64,
                reverse_timer: 0.0,
                post_reverse_lockout: 0.0,
                shift_cooldown: 0.0,
                shift_up_prev: 0.0,
                shift_down_prev: 0.0,
                nm_reserva: [1.0; 4],
                tire_temp: universe::tire_thermal::DEFAULT_AMBIENT,
                re_falhas: 0,
                re_desde_saida: 999.0,
                dano: 0.0,
                joy_tick: 0.0,
                learning_reward: 0.0,
                joy_streak: 0.0,
                edge_row6: 0.0,
                edge_row7: 0.0,
                phantom_d_min: 30.0,
                dist_wall_l: 40.0,
                dist_wall_r: 40.0,
                dist_wall_f: 50.0,
            }
        })
        .collect();

    let mut dist_acum: Vec<f64> = vec![0.0; n_seres];
    let mut prog_acum: Vec<f64> = vec![0.0; n_seres];
    let mut checkpoint: Vec<u32> = vec![0; n_seres];
    let mut prog_prev: Vec<f32> = (0..n_seres)
        .map(|i| world_ho.get_track_progress(states[i].phys.pos[0], states[i].phys.pos[2]))
        .collect();
    let mut pos_prev: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

    // Tick de aquecimento — mesmo contrato do laço de treino (JIT/compilação
    // descartada, não entra na fitness).
    if organism_taxi {
        ctx.start_organism(genomes, &held_stations, &held_city.as_ref().unwrap().traffic_lights(), n_seres, TaxiMode::Production).expect("held-out organism birth");
        ctx.taxi.as_mut().unwrap().configure_routes(&ctx.device, &world_ho, &held_stations).expect("held-out road routes");
        ctx.sample_organism(&mut states).expect("held-out initial state");
    } else {
        assert!(ctx.tick(&mut states, n_seres, 1.0 / 64.0, 0, kan_knots), "held-out warmup failed");
    }

    let dt = 1.0_f32 / 64.0;
    for tick_i in 0..n_ticks {
        ctx.force_organism_sample = tick_i + 1 == n_ticks;
        if !ctx.tick(&mut states, n_seres, dt, tick_i + 1, kan_knots) {
            panic!("held-out failed at tick {tick_i}; partial results are invalid");
        }
        if organism_taxi { continue; }
        // `ctx.tick()` já decrementa `life_left`/incrementa `ticks_alive` por conta
        // própria (mesmo caminho usado no treino) — este laço só ACUMULA fitness,
        // não mexe em vida.
        for i in 0..n_seres {
            let p1 = pos_of(&states[i].phys);
            if states[i].life_left > 0.0 {
                let dx = (p1[0] - pos_prev[i][0]) as f64;
                let dz = (p1[2] - pos_prev[i][2]) as f64;
                dist_acum[i] += (dx * dx + dz * dz).sqrt();
                let on_hole =
                    world_ho.get_voxel_at(p1[0], p1[2]) == universe::world::VoxelType::Hole;
                let state = &mut states[i];
                if survival::apply_hole_contact(
                    &mut state.life_left,
                    &mut state.hole_cooldown_ticks,
                    on_hole,
                    n_ticks.max(1) as f64,
                ) {
                    state.hole_hits = state.hole_hits.saturating_add(1);
                }

                let cur = world_ho.get_track_progress(p1[0], p1[2]);
                let ld = lap_dist as f32;
                let mut d = cur - prog_prev[i];
                if d < -ld * 0.5 {
                    d += ld;
                } else if d > ld * 0.5 {
                    d -= ld;
                }
                let avanco = d.clamp(-ld * 0.1, ld * 0.1) as f64;
                prog_acum[i] += avanco;
                states[i].learning_reward = (avanco * 64.0 / 30.0).clamp(-1.0, 1.0) as f32;
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
                    if dd < best_d {
                        best_d = dd;
                        melhor = idx;
                    }
                }
                states[i].wp_near = melhor;
                let dist_eixo = best_d.sqrt();
                let half = (track_width * 0.5).max(1.0);
                let prox = (1.0 - dist_eixo / half).clamp(0.0, 1.0) as f64;
                if avanco > 0.0 {
                    states[i].center_prog += avanco * prox;
                }

                // Setor SEMPRE uniforme (ver doc da função — held-out ignora
                // `--portal-curva` de propósito, pra manter a contagem de setores
                // comparável entre treino e held-out).
                let setor = sector_of(cur, ld);
                let esperado = checkpoint[i] % N_CHECKPOINTS.max(1);
                if setor == esperado {
                    checkpoint[i] += 1;
                }
            }
            pos_prev[i] = p1;
        }
    }

    let mut fitness_ho: Vec<f64> = match modo_fitness {
        "dist" => dist_acum,
        "prog" => prog_acum.iter().map(|p| p.max(0.0)).collect(),
        "centro" => (0..n_seres)
            .map(|i| {
                let ordered = objective_cost::ordered_progress_metres(
                    checkpoint[i],
                    prog_acum[i],
                    lap_dist,
                    N_CHECKPOINTS as usize,
                );
                objective_cost::centered_progress_metres(ordered, states[i].center_prog)
            })
            .collect(),
        "checkpoint" => (0..n_seres)
            .map(|i| {
                objective_cost::ordered_progress_metres(
                    checkpoint[i],
                    prog_acum[i],
                    lap_dist,
                    N_CHECKPOINTS as usize,
                )
            })
            .collect(),
        "checkpoint-legacy" => (0..n_seres)
            .map(|i| checkpoint[i] as f64 * lap_dist + prog_acum[i].max(0.0))
            .collect(),
        "centro-legacy" => (0..n_seres)
            .map(|i| checkpoint[i] as f64 * lap_dist + states[i].center_prog.max(0.0))
            .collect(),
        _ => unreachable!("fitness mode validated before GPU setup"),
    };

    if organism_taxi {
        ctx.sample_organism(&mut states).expect("held-out final state");
        fitness_ho = ctx.organism_samples.iter().map(universe::organism_scenario::fitness).collect();
    }
    ctx.organism = organism_saved.0;
    ctx.taxi = organism_saved.1;
    ctx.organism_feedback = organism_saved.2;
    ctx.organism_samples = organism_saved.3;
    ctx.traffic = organism_saved.4;
    ctx.traffic_samples = organism_saved.5;
    ctx.driving_diagnostics = saved_diagnostics;
    if !organism_taxi && (wall_cost > 0.0 || impact_cost > 0.0) {
        let sector_metres = lap_dist / N_CHECKPOINTS as f64;
        for (score, state) in fitness_ho.iter_mut().zip(&states) {
            *score = objective_cost::safety_adjusted_score(
                *score,
                state.wall_ticks,
                wall_cost,
                state.collision_energy_j,
                impact_cost,
                sector_metres,
            );
        }
    }

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
    // `regiao_l1`/`regiao_l2` (`--regions`, src/analysis/regioes.rs) andam em lockstep
    // com `l1_samples`/`l2_samples` — mesmo risco de contaminação, mesmo conserto.
    ctx.regiao_l1.clear();
    ctx.regiao_l2.clear();
    // Hostile world back exactly as the training generation left it, including the
    // per-generation telemetry counters: the held-out ticks ran with the mechanisms
    // suspended, so anything they added to those counters would be a number attributed
    // to a generation that did not produce it.
    ctx.stations = hostile_saved.0;
    ctx.predator = hostile_saved.1;
    ctx.thermal = hostile_saved.2;
    ctx.sensor_cost = hostile_saved.3;
    ctx.tel_dry_beings = telemetry_saved.0;
    ctx.tel_predator_ticks = telemetry_saved.1;
    ctx.tel_predator_life = telemetry_saved.2;
    ctx.tel_station_fuel = telemetry_saved.3;
    ctx.tel_tire_temp_sum = telemetry_saved.4;
    ctx.tel_tire_temp_n = telemetry_saved.5;
    ctx.hip_mem = hip_saved.0;
    ctx.hip_genes = hip_saved.1;
    ctx.hip_writer = hip_saved.2;

    let mediana = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let vivos: Vec<f64> = (0..n_seres)
        .filter(|&i| states[i].life_left > 0.0)
        .map(|i| fitness_ho[i])
        .collect();
    if vivos.is_empty() {
        eprintln!(
            "[held-out] ninguém sobreviveu na pista reservada — mediana calculada \
                    sobre TODOS os seres (não só vivos), número menos confiável."
        );
        mediana(fitness_ho)
    } else {
        mediana(vivos)
    }
}

// ── Ponto de entrada ──────────────────────────────────────────────────────────

/// Finds the highest-numbered `evolve_best_gen*.bin` in `dir`, or `None`.
///
/// This used to be written inline as `for g in 0..n_gens`, bounding the search by the
/// number of generations about to run instead of by what is on disk. With `--gens 1` it
/// only ever looked for `evolve_best_gen0000.bin`, so `--resume` on a directory holding
/// `evolve_best_gen0043.bin` found nothing and fell through to a RANDOM population — while
/// the header had already printed "resuming from checkpoint". Measured 2026-08-25: a day
/// of "champion" analysis ran on default genomes and only the τ telemetry gave it away
/// (p10 = median = p90 = 2.000 s, which is `DECAY_DEFAULT`, not an evolved bank).
///
/// Read the directory. Never bound this by `n_gens`.
fn latest_checkpoint(dir: &str) -> Option<(usize, String)> {
    let mut latest: Option<(usize, String)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("evolve_best_gen") else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".bin") else {
            continue;
        };
        let Ok(gen_id) = digits.parse::<usize>() else {
            continue;
        };
        if latest.as_ref().is_none_or(|(best, _)| gen_id > *best) {
            latest = Some((gen_id, entry.path().to_string_lossy().into_owned()));
        }
    }
    latest
}

#[cfg(test)]
mod checkpoint_tests {
    use super::{N_CHECKPOINTS, automatic_fuel_tank, latest_checkpoint};

    /// The regression: a checkpoint numbered far above `--gens` must still be found.
    #[test]
    fn finds_a_checkpoint_numbered_above_gens() {
        let dir = std::env::temp_dir().join(format!("uni_ckpt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_string_lossy().into_owned();

        assert!(latest_checkpoint(&d).is_none(), "empty dir must give None");

        for nome in ["evolve_best_gen0043.bin", "evolve_best_gen0007.bin"] {
            std::fs::write(dir.join(nome), b"x").unwrap();
        }
        // Files that must NOT be picked up.
        for nome in [
            "evolve_best_gen.bin",
            "evolve_best_genABCD.bin",
            "outra.bin",
        ] {
            std::fs::write(dir.join(nome), b"x").unwrap();
        }

        let (gen_id, path) = latest_checkpoint(&d).expect("must find gen 43");
        assert_eq!(
            gen_id, 43,
            "must take the HIGHEST generation, not the first listed"
        );
        assert!(
            path.ends_with("evolve_best_gen0043.bin"),
            "wrong path: {path}"
        );

        // The bug in one line: the old code searched `0..n_gens`, so with n_gens = 1 it
        // looked only at gen 0000 and missed this file entirely.
        assert!(gen_id >= 1, "43 is far above any plausible --gens 1");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_directory_gives_none() {
        assert!(latest_checkpoint("/definitely/not/a/directory/1234").is_none());
    }

    #[test]
    fn automatic_fuel_matches_the_actual_refill_interval() {
        let lap = 2_400.0;
        let sector_tank = automatic_fuel_tank(lap, 1.0, true);
        let lap_tank = automatic_fuel_tank(lap, 1.0, false);
        assert!((lap_tank / sector_tank - N_CHECKPOINTS as f32).abs() < 1e-5);
        assert!((lap_tank - (2_400.0 / 18.75) * 0.006).abs() < 1e-6);
    }
}

/// One line describing what the elite SELECTED on top of the fly map (`--fly-log-every`).
///
/// Prints what evolution can actually change — the bindings, the command pairs, the mask
/// density and the two gains — and nothing about the map itself, which is fixed. The first
/// bindings are listed as `channel->group`; a full list of up to 192 pairs would be a wall
/// of text, and the count plus the head is what a reader compares between generations.
fn log_fly_elite(gen_abs: usize, block: &[f32], layout: universe::fly_map::FlyLayout) {
    use universe::fly_map::{OFF_GAIN, OFF_OUT_PAIR, OFF_SCALE, bound_channels, mask_density};
    if block.len() != layout.block_len() {
        println!("  [fly-map] gen {gen_abs} elite: no fly block for this map");
        return;
    }
    let bound = bound_channels(block, layout);
    let head: Vec<String> = bound
        .iter()
        .take(8)
        .map(|(ch, g)| format!("{ch}->{g}"))
        .collect();
    let slot = |s: usize| -> String {
        let v = block[OFF_OUT_PAIR + s];
        if v < 0.0 {
            "none".to_string()
        } else {
            format!("{}", v as u32)
        }
    };
    println!(
        "  [fly-map] gen {gen_abs} elite: {} bound channels [{}{}] · out thr={}/{} str={}/{} \
         brk={}/{} · mask {:.1}% on · gain {:.3} · scale {:.3}",
        bound.len(),
        head.join(" "),
        if bound.len() > head.len() { " ..." } else { "" },
        slot(0),
        slot(1),
        slot(2),
        slot(3),
        slot(4),
        slot(5),
        mask_density(block, layout) * 100.0,
        block[OFF_GAIN],
        block[OFF_SCALE],
    );
}

/// Cheap lookup (no `VoxelWorld` build) of what index `idx` (either `gen_abs` or
/// `idx_curriculo`, same convention `pista_da_gen` uses) would train on: returns
/// `(display_seed, change_key)`. `mundos` (`--world`/`--world-file`) wins over
/// `pistas` (`--tracks`) when both are given — composes with `--switch-on-complete`
/// exactly like `--tracks` did, via the same rotation index.
///
/// `change_key` is what the caller compares against its previous value to decide
/// whether to reload (`world_seed_atual`/`world_key_atual`): plain `seed` for the
/// `--tracks` path, or `WorldSpec::fingerprint()` for `--world` — a raw seed
/// comparison would miss a change between two specs that share a `seed` but differ in,
/// say, `width`. Kept separate from `build_world` (below) so a generation whose track
/// did not change never pays for rebuilding the 320x320 voxel grid.
fn peek_world(
    pistas: &[u64],
    mundos: &[universe::world_spec::WorldSpec],
    idx: usize,
) -> (u64, u64) {
    if !mundos.is_empty() {
        let spec = spec_da_gen(mundos, idx);
        (spec.seed, spec.fingerprint())
    } else {
        let seed = pista_da_gen(pistas, idx);
        (seed, seed)
    }
}

/// Builds the `VoxelWorld` for training index `idx` — see `peek_world` for the
/// (cheap) rotation rule this follows. Returns `(world, display_seed, change_key)`.
fn resolve_world(
    pistas: &[u64],
    mundos: &[universe::world_spec::WorldSpec],
    idx: usize,
    track_width: f32,
) -> (VoxelWorld, u64, u64) {
    if !mundos.is_empty() {
        let spec = spec_da_gen(mundos, idx);
        (
            VoxelWorld::new_from_spec(&spec),
            spec.seed,
            spec.fingerprint(),
        )
    } else {
        let seed = pista_da_gen(pistas, idx);
        (VoxelWorld::new(seed, track_width), seed, seed)
    }
}

fn main() {
    // Production body defaults. The automatic gearbox is limited to third gear until
    // the curriculum proves that perception/control can safely use a higher ceiling.
    CAMBIO_MODO.store(1, Ordering::Relaxed);
    CAMBIO_TETO.store(3, Ordering::Relaxed);
    // Regional neuromodulation is useful only once slow body state changes the best
    // policy. Its earlier paired experiment tied without that regime, so keep it an
    // explicit mechanism instead of adding an inert layer to every controller.
    NEUROMOD_REGIOES.store(false, Ordering::Relaxed);
    // Collisions have a physical consequence by default. This is a body property,
    // not a driving policy: evolution remains free to discover how to avoid damage.
    DANO_ESCALA.store(1.0, Ordering::Relaxed);
    // Força ICD Vulkan da Radeon (evita SIGSEGV com múltiplos ICDs).
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe {
            std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
        }
    }

    // ── Parsing de args ────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut n_seres: usize = 60;
    // At the historical champion's measured 9-11 m/s, 8,000 ticks cannot cover the
    // 1,639 m validation lap. Twenty thousand ticks gives selection time to observe
    // at least one complete lap without requiring every command to override it.
    let mut n_ticks: usize = 20_000;
    let mut n_gens: usize = 100;
    // The native path now uses the same SlowBrain input semantics as the original
    // Universe body. `--slow-kinematics legacy` remains available for old replays.
    let mut canonical_slow_kinematics = true;
    let mut continuous_probes = true;
    let mut retomar: bool = false;
    // `--replay-every N`: write the leader replay only every N generations (and on the last
    // one). Rule 5 of CLAUDE.md: a 60-generation run at 30 000 ticks is ~58 MB per replay.
    let mut replay_every: usize = 1;
    let mut l1_vivos: usize = 0; // 0 = todos os 96 (sem ablacao)
    let mut retomar_mix: f32 = 0.85; // fracao do campeao preservada nos filhos
    let mut congela_l2_cegas: bool = true;
    let mut mascara_regioes: bool = true;
    let mut tau_semeado: bool = true;
    let mut observar_path: Option<String> = None;
    let mut gravar_onda_path: Option<String> = None;
    let mut trocar_ao_completar: usize = 0; // 0 = rotacao por geracao (comportamento atual)
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
    let mut out_dir = ".".to_string();
    let mut driving_diagnostics_on = false;
    let mut evolving_body = false;
    let mut organism_taxi = false;
    let mut taxi_reference_witness = false;
    let mut snapshot_every = 0usize;
    let mut resume_state: Option<String> = None;
    let mut lifetime_flags = 0u32;
    let mut evolve_learning = false;
    let mut adaptive_curriculum_on = false;
    let mut ecology_on = false;
    let mut ecology_founders: Option<usize> = None;
    let mut ecology_maturity: u64 = 640;
    let mut ecology_progress: f64 = 20.0;
    let mut vision_being: usize = 0;
    let mut vision_every: usize = 16;
    let mut ablate_spec: Option<String> = None;
    let mut ablate_being: usize = 0;
    let mut ablate_from_tick: Option<u32> = None;
    let mut erase_fast_memory = false;
    // Parâmetros evolutivos ajustáveis via CLI (padrão conservador para fine-tuning).
    // Para exploração inicial/escape de platô: use --sigma 0.15 --p-mutate 0.05
    let mut evo_sigma: f32 = 0.05;
    let mut evo_p_mutate: f32 = 0.02;
    // The three motor rows contain only 288 of ~23k fast weights. At the global 2%
    // rate they receive about six changed coefficients per child, which measured too
    // little exploration to cross the inherited negative brake logit. Motor rows use
    // their own default while all representation weights keep the conservative rate.
    let mut evo_motor_sigma: Option<f32> = Some(0.10);
    let mut evo_motor_p_mutate: Option<f32> = Some(0.10);
    // C1: rate/σ of the `hip` block (see `EvoParams::hip_p`); only used with `--hippocampus-sector`.
    let mut evo_hip_p: f32 = 1.0;
    let mut evo_hip_sigma: f32 = 0.2;
    // FLY MAP (docs/FLY-MAP-BRAIN.md). Every one of these is inert without `--fly-map`:
    // no path, no map, no buffers, no genome block, no draws in `evolve()`.
    let mut fly_map_path: Option<String> = None;
    let mut fly_spiking_on = false;
    let mut brain_control_direct = false;
    let mut fly_tau_steps: u32 = universe::gpu::fly_map::DEFAULT_TAU_STEPS;
    let mut evo_fly_p_bind: f32 = 0.05;
    let mut evo_fly_p_off: f32 = 0.05;
    let mut evo_fly_p_on: f32 = 0.015;
    let mut evo_fly_sigma: f32 = 0.1;
    let mut fly_log_every: usize = 0;
    let mut fly_spiking_trace_path: Option<String> = None;
    let mut fly_spiking_trace_every: usize = 64;
    let mut fly_spiking_trace_being: usize = 0;
    // `--fly-readout reflex` (docs/FLY-MAP-BRAIN.md "Reflex readout"): the anatomical
    // steering reflex measured in the connectome (asymmetry index swing +1.74/+1.92
    // rate model, +0.88 across the descending population, `neuromod-lab/PREDICOES.md`
    // 191v/193v) read straight off the map instead of evolved — `--fly-cruise-throttle`
    // is the host-supplied constant for the slot the map declares NONE (only throttle,
    // in the shipped export), `--fly-reflex-sign` the undocumented left/right convention
    // as an explicit experiment arm.
    let mut fly_cruise_throttle_v: f32 = 0.6;
    let mut fly_reflex_sign_v: f32 = 1.0;
    let mut fly_reflex_alpha_v: f32 = 0.0;
    let mut fly_reflex_gain_v: f32 = 1.0;
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
    let mut wall_cost: f32 = 2.0;
    // Metres of objective charged per kilojoule dissipated in a physical impact.
    // Unlike wall proximity, this signal grows quadratically with normal impact speed
    // through the kernel's energy term. One metre/kJ makes a typical 473 kJ collision
    // history worth ~10% of a 4.7 km run after the checkpoint unit correction.
    let mut impact_cost: f32 = 1.0;
    /// Cost curriculum (`--cost-gate-progress <metres>`), OFF by default.
    ///
    /// `--wall-cost` and `--impact-cost` are a finishing pressure: they rank a being that
    /// already drives above one that drives into walls. Applied from generation zero they
    /// rank against the only lineages that move at all, because at the start everybody
    /// crashes -- measured in B1, where seed 113 never crosses 1 000 m of median progress
    /// with the cost on and does cross it with the cost off (`docs/PREDICOES-CEREBRO.md`
    /// B1h/B2). With this flag the costs stay at zero until the population's MEDIAN
    /// uncosted fitness reaches the threshold, and apply from the next generation on.
    /// `None` keeps the historical behaviour bit for bit.
    let mut cost_gate_progress: Option<f64> = None;
    let mut cost_gate_open = false;
    let mut brain_skip: u32 = 1;
    /// Ticks de vida consumidos A MAIS por tick colado na parede. 0 = sem custo.
    let mut wall_life: i64 = 0;
    /// Ticks de vida consumidos A MAIS por tick PARADO (stagnation_ticks > 128). 0 = sem custo.
    let mut stag_life: i64 = 0;
    /// Todos os seres nascem no mesmo ponto. Torna a fitness comparável entre indivíduos
    /// — sem isso ela mede a largada, não o cérebro.
    let mut same_start = true;
    let mut start_formation = false;
    let mut largada_aleatoria = false;
    /// organism-taxi-v1: keep the being -> spawn slot map fixed across generations.
    let mut fixed_spawn_slots = false;
    /// organism-taxi-v1 navigation observation schema: 2 = dx/dz per 100 m, 3 = body-frame direction + compressed distance.
    let mut navigation_observation: u32 = 2;
    // organism school (docs/SCHOOL-LESSONS-20260922.md): one motor lesson judged per tick on the GPU.
    let mut school_lesson: Option<universe::gpu::school::Lesson> = None;
    // School mode; Stage 2 implements only `bench` (contacts and passengers off).
    let mut school_bench = true;
    // Stop-score shaping (docs/SCHOOL-STOP-SHAPING-20260922.md); off by default.
    let mut school_stop_shaping = false;
    let mut school_shared_start: Option<universe::organism_school::SharedStart> = None;
    // (candidate index, fold count) of the shared start of the current generation.
    let mut school_shared_pick: Option<(usize, usize)> = None;
    // Judged ticks per lesson attempt; defaults to the generation tick budget.
    let mut school_horizon: Option<u32> = None;
    // Number of simultaneous physical replicas used to estimate one genome's
    // school fitness. One is the historical compatibility arm.
    let mut school_eval_starts: usize = 1;
    // Stage 3 (docs/SCHOOL-INCENTIVES-20260922.md): curriculum over lessons, start folds,
    // online-reward composition and mid-life plasticity freeze.
    let mut school_curriculum: Option<&'static str> = None;
    let mut school_lessons: Vec<universe::gpu::school::Lesson> = Vec::new();
    let mut school_advance_p: f64 = 0.5;
    let mut school_advance_gens: usize = 2;
    let mut school_start_fold = universe::organism_school::LessonFold::All;
    let mut feedback_mode: u32 = 0;
    let mut plasticity_freeze_after: Option<usize> = None;
    let mut ruido_sigma: f32 = 0.0;
    let mut ruido_drop: f32 = 0.0;
    let mut clima_mu: f32 = 1.0;
    let mut clima_freq: f32 = 0.0;
    let mut vento_x: f32 = 0.0;
    let mut vento_z: f32 = 0.0;
    let mut fantasma_on = false;
    let mut escada_on = false;
    let mut piloto_formula_path: Option<String> = None;
    let mut piloto_forja_path: Option<String> = None;
    let mut portao_aleatorio: f32 = 0.0;
    let mut portao_semente: u64 = 0;
    /// Ticks de vida ganhos por METRO avançado. 0 = vida dada de graça (comportamento antigo).
    ///
    /// Medido: com vida fixa, a população converge para RASTEJAR — 2,2 m/s de média com
    /// 37,7 m/s disponíveis, e mortalidade 0,0%. Pior: com MAIS tempo eles ficam mais
    /// lentos (1,48 m/s em 24k ticks contra 2,96 m/s em 8k). A fitness é distância em
    /// tempo fixo, então devagar-e-seguro ganha de rápido-e-bate: não existe pressa.
    ///
    /// Ganhar vida avançando é a tese do projeto aplicada ao corpo: sobreviver EXIGE
    /// progredir. Não diz ao ser como dirigir nem para onde virar — diz que parar é morrer.
    let mut life_earn: f32 = 12.0;
    /// Largura da pista em metros. Estava fixa em 18,0 no código. A arqueologia de git
    /// mostrou que a versão que o Leonardo lembra rodava com 28 m; pista mais larga deixa
    /// carregar velocidade na curva, que é exatamente onde os seres perdem tempo hoje.
    let mut track_width: f32 = 28.0;
    /// `--fitness dist` volta ao hodômetro antigo. Padrão: progresso na pista.
    /// `checkpoint` (padrão) · `prog` · `dist`. Ver o `match` que monta a fitness.
    let mut modo_fitness = String::from("checkpoint");
    /// Alisamento do volante. 0,5 é o histórico; 1,0 remove a histerese.
    let mut steer_alpha: f32 = 0.5;
    // Signed arbitration preserves both directions of the fast motor signal. The
    // historical max(0, fast_throttle) erased its ability to oppose the slow drive;
    // `--throttle-arbitration legacy` remains available for checkpoint reproduction.
    let mut motor_head_mode: u32 = 0;
    let mut throttle_arbitration_mode: u32 = 1;
    let mut reverse_controller_mode = reverse_controller::ReverseMode::V2;
    // Fuel creates a dense survival requirement: reaching the next ordered sector
    // replenishes the small tank; staying stuck exhausts it. It never chooses an action.
    let mut fuel_on = true;
    let mut fuel_auto_k: Option<f32> = Some(1.5);
    let mut refuel_laps: u32 = 1;
    let mut fuel_tank: f32 = 1.0;
    let mut setor_paga = false;
    /// `--auditar-visao`: mede se os canais sensoriais respondem à pista real.
    let mut auditar_visao = false;
    // `--retina-range <m>`: how far the whole perceptual horizon reaches.
    //
    // Default `ALCANCE_PADRAO` (33.3 m), the value the retina's geometry already had —
    // omitting the flag changes not one uploaded bit (`visao::range_scale_of` pins the
    // factor to exactly 1.0 and both packers branch on it).
    //
    // Why it exists: U6/U10b/U11 of `docs/PREDICOES-CEREBRO.md` all failed to make the
    // second lap beatable, because on track 8005 at ~9 m/s nothing about the track is
    // hidden — the being SEES every curve in time, so there is nothing a per-sector
    // memory could know that perception does not already deliver. Shortening the
    // horizon is what puts information beyond it.
    let mut road_view_range_m: Option<f32> = None;
    let mut retina_range_m: f32 = universe::visao::ALCANCE_PADRAO;
    let mut portal_curva = false;
    let mut slow_steer_auth: f32 = 0.20;
    /// Accumulated-life ceiling in ticks. Zero follows the episode length when
    /// life earning is disabled, or the 3,840-tick ecological reserve otherwise.
    let mut life_cap_arg: i64 = 0;
    let mut evo_n_elite: usize = 3;
    let mut evo_tourney: usize = 3;
    // gen_offset: número de geração inicial para relatório (útil ao retomar).
    let mut gen_offset: usize = 0;
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
    let mut out_beta: f32 = universe::gpu::brain::OUT_BETA_DEFAULT;
    /// `--output-gain`: ganho aplicado a l2_raw/bypass na ETAPA 2 antes do tanh
    /// (P1.5, `docs/PLANO_EVOLUCAO.md`). Padrão = a constante 2.0 histórica —
    /// o A/B pendente compara 2.0 vs 1.0 vs 0.5 contra a saturação de L2
    /// (48/48 neurônios ativos em 97,4% dos ticks).
    let mut ganho_saida: f32 = universe::gpu::brain::OUT_GAIN_DEFAULT;
    /// `--norm-l2-by-deviation` / `--no-norm-l2-by-deviation`: segunda alavanca de
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
    let mut thr_fixo: f32 = 0.0;
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
    /// `--tracks "9001,9002,..."`: conjunto de pistas de TREINO, rotacionadas
    /// deterministicamente por geração (`pista_da_gen`, gen_abs % len). Vazio
    /// (default) = validated starter track 8005. Pass `--tracks 9001` explicitly
    /// only when reproducing the historical checkpoint trained on that seed.
    let mut pistas: Vec<u64> = Vec::new();
    /// `--world <spec>` / `--world-file <path.json>` (`universe::world_spec`):
    /// parametric track requests (seed + radius/curves/width/holes/length),
    /// rotated by the SAME index `pistas` uses (`spec_da_gen`, `pistas.rs`). When
    /// non-empty this list wins over `pistas` — see `resolve_world` below. Order:
    /// `--world-file` entries first, then each `--world` in the order given.
    let mut mundos: Vec<universe::world_spec::WorldSpec> = Vec::new();
    /// `--track-held-out <seed>`: pista RESERVADA, nunca sorteada por `pista_da_gen`
    /// pro treino — só para MEDIR generalização (ver bloco `[generalizacao]`).
    /// `None` (padrão) desliga a medição inteira: nenhum tick extra roda.
    let mut pista_held_out: Option<u64> = None;
    /// `--held-out-cada <K>`: mede held-out a cada K gerações (padrão 5). Só lido
    /// quando `pista_held_out` está setada.
    let mut held_out_cada: usize = 5;
    {
        let mut i = 1;
        while i < args.len() {
            // Every arm below matches the CANONICAL English name, once, from
            // `cli_flags::FLAGS`. `resolve` maps a deprecated spelling onto it and warns
            // once per alias, so a battery request already in the queue keeps working
            // without this loop carrying two literals per flag. Reject unknown
            // arguments so experimental configurations cannot silently change.
            let raw = args[i].as_str();
            let arg = cli_flags::resolve(raw).map_or(raw, |f| f.name);
            match arg {
                "--driving-diagnostics" => {
                    driving_diagnostics_on = true;
                }
                "--school-stop-shaping" => {
                    school_stop_shaping = true;
                }
                "--school-shared-start" => {
                    i += 1;
                    school_shared_start = args.get(i).and_then(|a| universe::organism_school::SharedStart::parse(a));
                    if school_shared_start.is_none() {
                        eprintln!("--school-shared-start requires rotate or fixed");
                        std::process::exit(2);
                    }
                }
                "--erase-fast-memory" => {
                    erase_fast_memory = true;
                }
                "--vision-being" => {
                    i += 1;
                    vision_being = args.get(i).and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                        eprintln!("ERROR --vision-being requires a nonnegative integer");
                        std::process::exit(1)
                    });
                }
                "--vision-every" => {
                    i += 1;
                    vision_every = args
                        .get(i)
                        .and_then(|v| v.parse().ok())
                        .filter(|v| *v > 0)
                        .unwrap_or_else(|| {
                            eprintln!("ERROR --vision-every requires a positive integer");
                            std::process::exit(1)
                        });
                }
                "--ablate" => {
                    i += 1;
                    ablate_spec = Some(args.get(i).cloned().unwrap_or_else(|| {
                        eprintln!("ERROR --ablate requires cfc:N,l1:N,l2:N targets");
                        std::process::exit(2)
                    }));
                }
                "--ablate-being" => {
                    i += 1;
                    ablate_being = args.get(i).and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                        eprintln!("ERROR --ablate-being requires a nonnegative integer");
                        std::process::exit(2)
                    });
                }
                "--ablate-from-tick" => {
                    i += 1;
                    ablate_from_tick =
                        Some(args.get(i).and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                            eprintln!("ERROR --ablate-from-tick requires a u32 tick");
                            std::process::exit(2)
                        }));
                }
                "--beings" => {
                    i += 1;
                    if i < args.len() {
                        n_seres = args[i].parse().unwrap_or(60);
                    }
                }
                "--ticks" => {
                    i += 1;
                    if i < args.len() {
                        n_ticks = args[i].parse().unwrap_or(8_000);
                    }
                }
                "--gens" => {
                    i += 1;
                    if i < args.len() {
                        n_gens = args[i].parse().unwrap_or(100);
                    }
                }
                "--sigma" => {
                    i += 1;
                    if i < args.len() {
                        evo_sigma = args[i].parse().unwrap_or(0.05);
                    }
                }
                "--p-mutate" => {
                    i += 1;
                    if i < args.len() {
                        evo_p_mutate = args[i].parse().unwrap_or(0.02);
                    }
                }
                "--motor-p-mutate" => {
                    i += 1;
                    if i < args.len() {
                        evo_motor_p_mutate = args[i].parse().ok();
                    }
                }
                "--motor-sigma" => {
                    i += 1;
                    if i < args.len() {
                        evo_motor_sigma = args[i].parse().ok();
                    }
                }
                "--hip-p" => {
                    i += 1;
                    if i < args.len() {
                        evo_hip_p = args[i].parse().unwrap_or(1.0);
                    }
                }
                "--hip-sigma" => {
                    i += 1;
                    if i < args.len() {
                        evo_hip_sigma = args[i].parse().unwrap_or(0.2);
                    }
                }
                "--no-slow" => {
                    SLOW_ABLATED.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                // Curva de capacidade: quantos dos 96 neuronios de L1 ficam vivos.
                // O PR mede 6,7 dimensoes em uso; esta flag testa se isso se traduz em
                // metros — cortar ate onde o desempenho aguentar.
                "--l1-alive" => {
                    i += 1;
                    if i < args.len() {
                        l1_vivos = args[i].parse().unwrap_or(0);
                    }
                }
                "--resume-mix" => {
                    i += 1;
                    if i < args.len() {
                        retomar_mix = args[i].parse().unwrap_or(0.85);
                    }
                }
                "--switch-on-complete" => {
                    i += 1;
                    if i < args.len() {
                        trocar_ao_completar = args[i].parse().unwrap_or(1);
                    }
                }
                // P1.8: piso de velocidade mediana para a rota alternativa de promoção
                // do currículo (ver doc de `curriculo_veloc_min` acima).
                "--curriculum-min-speed" => {
                    i += 1;
                    if i < args.len() {
                        curriculo_veloc_min = args[i].parse().unwrap_or(curriculo_veloc_min);
                    }
                }
                // Banco de τ log-uniforme em [25 ms, 4 s] em vez do τ = 2,0 s uniforme.
                // Sem a flag o comportamento é o histórico — é o braço de controle.
                "--tau-hetero" => {
                    tau_hetero = true;
                }
                // Prediction 186: which PARAMETRISATION builds the decay-rate bank that
                // is uploaded, without ever rewriting `Genome::tau` (so checkpoints and
                // `--resume` cross modes). `free` is the default and is today's
                // behaviour bit for bit. See `universe::tau_bank`. The three values had
                // Portuguese spellings until 2026-09-12; `cli_flags::resolve_value`
                // still accepts them, once per run, with a warning.
                "--tau-bank" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!(
                            "[ERROR] --tau-bank requires a value: free | log-fixed | spectral.\n\
                             [ERROR] `free` is the default (96 free genes, historical behaviour)."
                        );
                        std::process::exit(1);
                    }
                    let given = args[i].as_str();
                    let value = cli_flags::resolve_value(arg, given).unwrap_or(given);
                    match universe::tau_bank::TauBankMode::from_flag(value) {
                        Some(m) => set_tau_bank_mode(m),
                        None => {
                            eprintln!(
                                "[ERROR] --tau-bank {given}: unknown mode.\n\
                                 [ERROR] Valid values are: free | log-fixed | spectral.\n\
                                 [ERROR] Refusing to fall back to `free` — a run that\n\
                                 [ERROR] silently swaps the arm under test reports plausible\n\
                                 [ERROR] numbers for a bank you did not ask for."
                            );
                            std::process::exit(1);
                        }
                    }
                }
                // P0.5: inicialização espectral de L1 vem LIGADA por padrão (genoma
                // nasce com ganho espectral ≈ 1.0 em vez do ~7.0 cru medido com
                // σ=0.30 i.i.d. — ver `gpu::evolution::Genome::random`). Esta flag
                // DESLIGA para rodar o braço de controle do A/B.
                "--no-spectral-init" => {
                    universe::gpu::evolution::SPECTRAL_INIT_ON.store(false, Ordering::Relaxed);
                }
                "--driving-reflex-birth" => {
                    universe::gpu::evolution::DRIVING_REFLEX_BIRTH_ON
                        .store(true, Ordering::Relaxed);
                }
                "--no-driving-reflex-birth" => {
                    universe::gpu::evolution::DRIVING_REFLEX_BIRTH_ON
                        .store(false, Ordering::Relaxed);
                }
                // FASE SENSORES (degrau 1): liga a evolução de FOV/alcances/lookaheads de
                // curvatura (ver doc de `SENSORS_EVOLVE_ON`). Padrão DESLIGADA — sem esta
                // flag, todo genoma nasce e permanece em `SENSOR_DEFAULTS` (os hardcodes de
                // hoje), protegendo qualquer bateria em curso que não passe a flag nova.
                "--evolving-sensors" => {
                    universe::gpu::evolution::SENSORS_EVOLVE_ON.store(true, Ordering::Relaxed);
                }
                "--snapshot-every" => {
                    i += 1;
                    snapshot_every =
                        args.get(i).and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                            eprintln!("invalid --snapshot-every");
                            std::process::exit(2)
                        });
                }
                "--resume-state" => {
                    i += 1;
                    resume_state = Some(args.get(i).expect("--resume-state requires path").clone());
                }
                "--taxi-reference-witness" => {
                    taxi_reference_witness = true;
                }
                "--scenario" => {
                    i += 1;
                    organism_taxi = match args.get(i).map(String::as_str) {
                        Some("organism-taxi-v1") => true,
                        Some("race-v1") => false,
                        _ => {
                            eprintln!("--scenario requires race-v1 or organism-taxi-v1");
                            std::process::exit(2);
                        }
                    };
                }
                "--ecology" => {
                    ecology_on = true;
                }
                "--ecology-founders" => {
                    i += 1;
                    ecology_founders = Some(args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0));
                }
                "--ecology-maturity-ticks" => {
                    i += 1;
                    ecology_maturity = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "--ecology-min-progress" => {
                    i += 1;
                    ecology_progress = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(f64::NAN);
                }
                "--plasticity" => {
                    lifetime_flags |= 1;
                }
                "--development" => {
                    lifetime_flags |= 2;
                }
                "--evolving-topology" => {
                    lifetime_flags |= 4;
                }
                "--evolve-learning" => {
                    evolve_learning = true;
                    lifetime_flags |= 1;
                }
                "--adaptive-curriculum" => {
                    adaptive_curriculum_on = true;
                }
                "--evolving-body" => {
                    evolving_body = true;
                }
                "--fixed-body" => {
                    evolving_body = false;
                }
                // Força de volta ao braço de controle (já é o padrão) — nome literal pedido
                // pela tarefa. Vence se vier DEPOIS de `--evolving-sensors` na linha de
                // comando, igual `--sem-lento`/`--sem-init-espectral` para outros experimentos.
                "--fixed-sensors" => {
                    universe::gpu::evolution::SENSORS_EVOLVE_ON.store(false, Ordering::Relaxed);
                }
                // P1.7 (docs/PLANO_EVOLUCAO.md) — braço de controle do cérebro
                // assíncrono, um por mecanismo. Cada um CONGELA a mutação do campo
                // (atomic em `gpu::evolution`) E força o shader a ignorar o que o
                // genoma carregar (`BrainGpu::*_off`) — ver a doc das variáveis
                // `sem_*` acima. Sem nenhuma destas flags (padrão), os três
                // mecanismos mutam livremente a partir do neutro de nascença.
                "--no-neuron-memory" => {
                    sem_memoria_neuronal = true;
                    universe::gpu::evolution::NEURON_MEM_MUTATE_ON.store(false, Ordering::Relaxed);
                }
                "--no-neuron-period" => {
                    sem_periodo_neuronal = true;
                    universe::gpu::evolution::NEURON_PERIOD_MUTATE_ON
                        .store(false, Ordering::Relaxed);
                }
                "--no-neuron-threshold" => {
                    sem_limiar_neuronal = true;
                    universe::gpu::evolution::NEURON_THRESHOLD_MUTATE_ON
                        .store(false, Ordering::Relaxed);
                }
                "--no-heterogeneous-activation" => {
                    sem_ativacao_heterogenea = true;
                    universe::gpu::evolution::NEURON_ACTIVATION_MUTATE_ON
                        .store(false, Ordering::Relaxed);
                }

                // C1 (docs/PRD-CEREBRO-NEUROMOD.md §C1) — sector hippocampus. Three
                // flags, three questions; see the doc of each static in `statics.rs`.
                //
                // `--hippocampus-sector` is also what turns the MUTATION of `Genome::hip` on.
                // It has to: that block is the last consumer of `rng` in `evolve()`, so a
                // run that does not ask for the hippocampus must not draw from it at all,
                // or every other field of every being after the first would stop matching
                // a run of the previous binary (see `HIPPOCAMPUS_MUTATE_ON`).
                // The third flag below is the control arm, and it wins if it comes AFTER
                // the first one on the command line — the same precedence rule as
                // `--fixed-sensors` over `--evolving-sensors`.
                //
                // check-language: off -- the three flag NAMES below are Portuguese, like
                // every other CLI flag this binary already accepts (`--no-slow`,
                // `--no-neuron-memory`). The PRD names them; renaming them here would
                // break the battery scripts of C4. Only the literals are exempted.
                "--hippocampus-sector" => {
                    HIPPOCAMPUS_ON.store(true, Ordering::Relaxed);
                    universe::gpu::evolution::HIPPOCAMPUS_MUTATE_ON.store(true, Ordering::Relaxed);
                }
                "--hippocampus-erase" => {
                    HIPPOCAMPUS_ERASE.store(true, Ordering::Relaxed);
                }
                "--no-hippocampus" => {
                    HIPPOCAMPUS_DISABLED.store(true, Ordering::Relaxed);
                    universe::gpu::evolution::HIPPOCAMPUS_MUTATE_ON.store(false, Ordering::Relaxed);
                }
                // check-language: on
                // Cobra o recuo na fitness `centro` (ver a acumulação de `center_prog`).
                "--symmetric-center" => {
                    centro_simetrico = true;
                }
                // Força da fadiga de SAÍDA. 0,85 (padrão) faz comando sustentado sair a
                // 15% da intenção em ~1 s; 0,0 desliga. Ver `BrainParams::out_beta`.
                "--out-beta" => {
                    i += 1;
                    if i < args.len() {
                        out_beta = args[i].parse().unwrap_or(out_beta);
                    }
                }
                // P1.5: ganho de saída (era a constante 2.0 hardcoded na ETAPA 2).
                // Ver `BrainParams::out_gain` e o comentário de `ganho_saida` acima.
                "--output-gain" => {
                    i += 1;
                    if i < args.len() {
                        ganho_saida = args[i].parse().unwrap_or(ganho_saida);
                    }
                }
                // P1.5/P1.8: normaliza l2_raw pelo desvio-padrão dos 48 valores do ser
                // antes do ganho/tanh. LIGADA por padrão desde P1.8 — esta flag vira
                // no-op explícito (mantida para não quebrar scripts existentes).
                "--norm-l2-by-deviation" => {
                    norm_l2_por_desvio = true;
                }
                // Braço de controle: desliga a normalização e volta ao comportamento
                // histórico pré-P1.8 (l2_std fixo em 1.0, divisão é no-op no shader).
                "--no-norm-l2-by-deviation" => {
                    norm_l2_por_desvio = false;
                }
                // Piloto fantasma: ganho do controlador geométrico. 0 = cérebro no comando.
                "--pilot" => {
                    i += 1;
                    if i < args.len() {
                        piloto = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--pilot-thr" => {
                    i += 1;
                    if i < args.len() {
                        piloto_thr = args[i].parse().unwrap_or(0.45);
                    }
                }
                "--steer-fixed" => {
                    i += 1;
                    if i < args.len() {
                        steer_fixo = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--steer-scale" => {
                    i += 1;
                    if i < args.len() {
                        steer_escala = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--pilot-sensor" => {
                    i += 1;
                    if i < args.len() {
                        piloto_sensor = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--thr-fixed" => {
                    i += 1;
                    if i < args.len() {
                        thr_fixo = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--seed" => {
                    i += 1;
                    if i < args.len() {
                        pop_seed = args[i].parse().unwrap_or(0);
                    }
                }
                "--wall-cost" => {
                    i += 1;
                    if i < args.len() {
                        wall_cost = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--impact-cost" => {
                    i += 1;
                    if i < args.len() {
                        impact_cost = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--cost-gate-progress" => {
                    i += 1;
                    if i < args.len() {
                        cost_gate_progress = args[i].parse::<f64>().ok().filter(|m| *m > 0.0);
                    }
                }
                "--brain-skip" => {
                    i += 1;
                    if i < args.len() {
                        brain_skip = args[i].parse().unwrap_or(1).max(1);
                    }
                }
                "--kernel-profile" => {
                    i += 1;
                    if i < args.len() {
                        // 0 = off (default). Prints per-kernel GPU milliseconds every N
                        // ticks — see `statics::PERFIL_KERNEL` and `context.rs::tick()`.
                        PERFIL_KERNEL.store(args[i].parse().unwrap_or(0), Ordering::Relaxed);
                    }
                }
                "--wall-life" => {
                    i += 1;
                    if i < args.len() {
                        wall_life = args[i].parse().unwrap_or(0);
                    }
                }
                // FÍSICA 1: vida/tick queimada por girar acima de 90% do RPM (0=off).
                // E-REG1: λ do custo de fiação por distância (0=off). Ver statics.rs.
                "--wiring-tail" => {
                    i += 1;
                    if i < args.len() {
                        FIACAO_CAUDA.store(args[i].parse().unwrap_or(0), Ordering::Relaxed);
                    }
                }
                "--wiring-cost" => {
                    i += 1;
                    if i < args.len() {
                        CUSTO_FIACAO.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--rpm-cost" => {
                    i += 1;
                    if i < args.len() {
                        CUSTO_RPM.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--stag-life-drain" => {
                    i += 1;
                    if i < args.len() {
                        stag_life = args[i].parse().unwrap_or(0);
                    }
                }
                "--same-start" => {
                    same_start = true;
                }
                "--fixed-spawn-slots" => {
                    fixed_spawn_slots = true;
                }
                "--school" => {
                    i += 1;
                    school_lesson = Some(
                        args.get(i)
                            .and_then(|a| universe::gpu::school::Lesson::parse(a))
                            .unwrap_or_else(|| {
                                eprintln!("--school requires move, stop, turn_near, turn_far, approach_passenger, pickup, navigate_with_passenger or delivery");
                                std::process::exit(2);
                            }),
                    );
                }
                "--school-mode" => {
                    i += 1;
                    school_bench = match args.get(i).map(String::as_str) {
                        Some("bench") => true,
                        // life: the judge observes the lesson while passengers and taxi-taxi
                        // contacts stay on; fitness remains the taxi fitness.
                        Some("life") => false,
                        _ => {
                            eprintln!("--school-mode requires bench or life");
                            std::process::exit(2);
                        }
                    };
                }
                "--school-curriculum" => {
                    i += 1;
                    school_curriculum = match args.get(i).map(String::as_str) {
                        Some("adaptive") => Some("adaptive"),
                        Some("uniform") => Some("uniform"),
                        _ => {
                            eprintln!("--school-curriculum requires adaptive or uniform");
                            std::process::exit(2);
                        }
                    };
                }
                "--school-lessons" => {
                    i += 1;
                    school_lessons = args
                        .get(i)
                        .map(|list| list.split(',').map(str::trim).filter(|s| !s.is_empty()).collect::<Vec<_>>())
                        .unwrap_or_default()
                        .iter()
                        .map(|name| {
                            universe::gpu::school::Lesson::parse(name).unwrap_or_else(|| {
                                eprintln!("--school-lessons: unknown lesson {name}");
                                std::process::exit(2);
                            })
                        })
                        .collect();
                    if school_lessons.is_empty() {
                        eprintln!("--school-lessons requires a comma-separated list of lessons");
                        std::process::exit(2);
                    }
                }
                "--school-advance-p" => {
                    i += 1;
                    school_advance_p = args.get(i).and_then(|a| a.parse().ok()).filter(|p: &f64| (0.0..=1.0).contains(p)).unwrap_or_else(|| {
                        eprintln!("--school-advance-p requires a probability in 0..=1");
                        std::process::exit(2);
                    });
                }
                "--school-advance-gens" => {
                    i += 1;
                    school_advance_gens = args.get(i).and_then(|a| a.parse().ok()).filter(|g| *g > 0).unwrap_or_else(|| {
                        eprintln!("--school-advance-gens requires a positive generation count");
                        std::process::exit(2);
                    });
                }
                "--school-start-fold" => {
                    i += 1;
                    school_start_fold = args.get(i).and_then(|a| universe::organism_school::LessonFold::parse(a)).unwrap_or_else(|| {
                        eprintln!("--school-start-fold requires all, train, validation or test");
                        std::process::exit(2);
                    });
                }
                "--feedback-mode" => {
                    i += 1;
                    feedback_mode = match args.get(i).map(String::as_str) {
                        Some("legacy") => 0,
                        Some("symmetric") => 1,
                        Some("progress") => 2,
                        _ => {
                            eprintln!("--feedback-mode requires legacy, symmetric or progress");
                            std::process::exit(2);
                        }
                    };
                }
                "--plasticity-freeze-after" => {
                    i += 1;
                    plasticity_freeze_after = Some(args.get(i).and_then(|a| a.parse().ok()).unwrap_or_else(|| {
                        eprintln!("--plasticity-freeze-after requires a tick index");
                        std::process::exit(2);
                    }));
                }
                "--school-horizon" => {
                    i += 1;
                    school_horizon = Some(args.get(i).and_then(|a| a.parse().ok()).filter(|h| *h > 0).unwrap_or_else(|| {
                        eprintln!("--school-horizon requires a positive tick count");
                        std::process::exit(2);
                    }));
                }
                "--school-eval-starts" => {
                    i += 1;
                    school_eval_starts = args
                        .get(i)
                        .and_then(|a| a.parse().ok())
                        .filter(|starts| *starts > 0)
                        .unwrap_or_else(|| {
                            eprintln!("--school-eval-starts requires a positive count");
                            std::process::exit(2);
                        });
                }
                "--navigation-observation" => {
                    i += 1;
                    navigation_observation = match args.get(i).map(String::as_str) {
                        Some("v2") => 2,
                        Some("v3") => 3,
                        _ => {
                            eprintln!("--navigation-observation requires v2 or v3");
                            std::process::exit(2);
                        }
                    };
                    NAVIGATION_OBSERVATION.store(navigation_observation, Ordering::Relaxed);
                }
                "--social-sensors" => {
                    SOCIAL_SENSORS_ON.store(true, Ordering::Relaxed);
                }
                "--start-formation" => {
                    start_formation = true;
                    same_start = true;
                }
                // D2 (ESCADA_DIFICULDADE.md): largada num waypoint ALEATÓRIO por geração
                // (o MESMO para todos os seres da gen — comparável; derivado da seed da
                // gen). Mata a decoração de traçado sem reintroduzir o ruído de largada.
                "--random-start" => {
                    largada_aleatoria = true;
                    same_start = true;
                }
                // D4: dano oculto (0=off; 1=escala padrão). Ver ESCADA_DIFICULDADE.md.
                "--damage" => {
                    i += 1;
                    if i < args.len() {
                        DANO_ESCALA.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                // D3: ruído sensorial (σ em unidades i8; 6 ≈ 5% da faixa) + prob de falha.
                "--sensory-noise" => {
                    i += 1;
                    if i < args.len() {
                        ruido_sigma = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--dropout-noise" => {
                    i += 1;
                    if i < args.len() {
                        ruido_drop = args[i].parse().unwrap_or(0.0);
                    }
                }
                // D5: clima — μ base (<1 = escorregadio), freq espacial, vento N.
                "--weather" => {
                    i += 1;
                    if i < args.len() {
                        clima_mu = args[i].parse().unwrap_or(1.0);
                    }
                }
                "--weather-freq" => {
                    i += 1;
                    if i < args.len() {
                        clima_freq = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--wind" => {
                    i += 1;
                    if i + 1 < args.len() {
                        vento_x = args[i].parse().unwrap_or(0.0);
                        i += 1;
                        vento_z = args[i].parse().unwrap_or(0.0);
                    }
                }
                // D6: fantasma do campeão anterior (bônus de vida por estar à frente).
                "--ghost" => {
                    fantasma_on = true;
                }
                // ESCADA completa (controlador automático de dificuldade; ver escada.rs).
                "--ladder" => {
                    escada_on = true;
                }
                // ── HOSTILE WORLD (BODY PHASE, 2026-08-17) ───────────────────
                //
                // INTEROCEPTION — turns the BODY BLOCK on: eight CPU-computed values
                // per being per tick reach the sensory wave (registry
                // `perception::channels::BODY_SLOTS`). Off by default: it changes the
                // input distribution of ch98/ch114-117, so an existing champion needs
                // a paired A/B, not faith (CLAUDE.md rule 3, same contract as
                // `--real-peripheral-vision` above).
                //
                // Two of the eight slots are SOURCE MOVES that carry the same number
                // the old path carried (ch70 with the thermal model off, ch71 always),
                // so the flag's real content is: OwnFuel becomes visible, and the five
                // measured-dead channels stop being constant. That is why it deserves
                // its own flag instead of riding along with a world pressure.
                "--interoception" => {
                    universe::gpu::perception::BODY_CHANNELS_ON.store(true, Ordering::Relaxed);
                }
                // D7 — scarce resource: M energy stations, M < N beings.
                "--resource-scarce" => {
                    i += 1;
                    if i < args.len() {
                        STATIONS_N.store(args[i].parse().unwrap_or(0), Ordering::Relaxed);
                    }
                }
                "--resource-load" => {
                    i += 1;
                    if i < args.len() {
                        STATION_CAPACITY.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--resource-regen" => {
                    i += 1;
                    if i < args.len() {
                        STATION_REGEN.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--resource-radius" => {
                    i += 1;
                    if i < args.len() {
                        STATION_RADIUS_M.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--resource-transfer" => {
                    i += 1;
                    if i < args.len() {
                        STATION_TRANSFER.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                // PREDATOR — dose is a FRACTION of the measured typical speed, so the
                // same number means the same pressure on every track.
                "--predator" => {
                    i += 1;
                    if i < args.len() {
                        PREDATOR_SPEED_FRAC
                            .store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--predator-drain" => {
                    i += 1;
                    if i < args.len() {
                        PREDATOR_DRAIN.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--predator-radius" => {
                    i += 1;
                    if i < args.len() {
                        PREDATOR_RADIUS_M.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--predator-perception" => {
                    i += 1;
                    if i < args.len() {
                        PREDATOR_PERCEPT_M.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--predator-grace" => {
                    i += 1;
                    if i < args.len() {
                        PREDATOR_GRACE_TICKS.store(args[i].parse().unwrap_or(0), Ordering::Relaxed);
                    }
                }
                "--predator-spawn" => {
                    i += 1;
                    if i < args.len() {
                        PREDATOR_SPAWN_BACK_M
                            .store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                // TYRE THERMAL — dose = depth of the grip curve; DC gain re-normalised
                // to 1.0 at every dose, so a bigger dose is a wider SWING, never less
                // grip on average (see TIRE_THERMAL_GAIN in statics.rs).
                "--tire-thermal" => {
                    i += 1;
                    if i < args.len() {
                        TIRE_THERMAL_GAIN.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                // D8 — perception costs energy: a per-family subscription paid in life.
                "--sensor-cost" => {
                    i += 1;
                    if i < args.len() {
                        SENSOR_COST.store(args[i].parse().unwrap_or(0.0), Ordering::Relaxed);
                    }
                }
                "--sensor-cost-k" => {
                    i += 1;
                    if i < args.len() {
                        SENSOR_COST_K.store(args[i].parse().unwrap_or(4), Ordering::Relaxed);
                    }
                }
                "--sensor-cost-cap" => {
                    i += 1;
                    if i < args.len() {
                        SENSOR_COST_CAP.store(args[i].parse().unwrap_or(2.0), Ordering::Relaxed);
                    }
                }
                // PILOTO POR FÓRMULA (PILOTO_POR_FORMULA.md): ser 0 dirigido pelo
                // controlador ridge destilado do replay do campeão.
                "--pilot-formula" => {
                    i += 1;
                    if i < args.len() {
                        piloto_formula_path = Some(args[i].clone());
                    }
                }
                // PILOTO v2 (não-linear, base pirâmide + arctanh na saída). Erra 31%
                // menos no held-out temporal que o linear (0,179 vs 0,261) — a pergunta
                // que este flag responde é se isso passa do limiar em que o erro deixa
                // de acumular em malha fechada.
                "--pilot-forge" => {
                    i += 1;
                    if i < args.len() {
                        piloto_forja_path = Some(args[i].clone());
                    }
                }
                // Controle do portão: abstenção ALEATÓRIA na mesma dose (ver
                // piloto_linear.rs). Responde "o portão escolhe bem ou qualquer
                // intervenção do cérebro na mesma taxa já bastaria?".
                "--random-gate" => {
                    i += 1;
                    if i < args.len() {
                        portao_aleatorio = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--gate-seed" => {
                    i += 1;
                    if i < args.len() {
                        portao_semente = args[i].parse().unwrap_or(0);
                    }
                }
                "--life-earn" => {
                    i += 1;
                    if i < args.len() {
                        life_earn = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--width" => {
                    i += 1;
                    if i < args.len() {
                        track_width = args[i].parse().unwrap_or(18.0);
                    }
                }
                "--fitness" => {
                    i += 1;
                    if i < args.len() {
                        modo_fitness = args[i].clone();
                    }
                }
                "--steer-alpha" => {
                    i += 1;
                    if i < args.len() {
                        steer_alpha = args[i].parse().unwrap_or(0.5);
                    }
                }
                "--motor-head" => {
                    i += 1;
                    motor_head_mode = match args.get(i).map(String::as_str) {
                        Some("spline") => 0,
                        Some("linear") => 1,
                        _ => {
                            eprintln!("ERROR --motor-head requires spline or linear");
                            std::process::exit(1)
                        }
                    };
                }
                "--throttle-arbitration" => {
                    i += 1;
                    throttle_arbitration_mode = match args.get(i).map(String::as_str) {
                        Some("legacy") => 0,
                        Some("signed") => 1,
                        _ => {
                            eprintln!("ERROR --throttle-arbitration requires legacy or signed");
                            std::process::exit(1)
                        }
                    };
                }
                "--reverse-controller-v2" => {
                    reverse_controller_mode = reverse_controller::ReverseMode::V2;
                }
                "--legacy-reverse-controller" => {
                    reverse_controller_mode = reverse_controller::ReverseMode::Legacy;
                }
                "--slow-kinematics" => {
                    i += 1;
                    canonical_slow_kinematics = match args.get(i).map(String::as_str) {
                        Some("legacy") => false,
                        Some("canonical") => true,
                        _ => {
                            eprintln!("ERROR --slow-kinematics requires legacy or canonical");
                            std::process::exit(1)
                        }
                    };
                }
                "--wall-probes" => {
                    i += 1;
                    continuous_probes = match args.get(i).map(String::as_str) {
                        Some("legacy") => false,
                        Some("continuous") => true,
                        _ => {
                            eprintln!("ERROR --wall-probes requires legacy or continuous");
                            std::process::exit(1)
                        }
                    };
                }
                // Regra de três do combustível: tanque = f(volta da pista) × k. Ver statics.rs.
                "--fuel-auto" => {
                    i += 1;
                    if i < args.len() {
                        fuel_auto_k = Some(args[i].parse().unwrap_or(0.0));
                        fuel_on = true;
                    }
                }
                "--fuel" => {
                    fuel_on = true;
                }
                "--fuel-tank" => {
                    i += 1;
                    if i < args.len() {
                        fuel_tank = args[i].parse().unwrap_or(1.0);
                        fuel_auto_k = None;
                        fuel_on = true;
                    }
                }
                "--no-fuel" => {
                    fuel_on = false;
                    fuel_auto_k = None;
                }
                "--refuel-lap-only" => {
                    setor_paga = false;
                }
                "--audit-vision" => {
                    auditar_visao = true;
                }
                // `--retina-beings` (docs/RETINA-BEINGS.md §2): the retina also
                // intersects each ray with every OTHER being's body. Off by default — it
                // changes the input distribution of every existing champion (CLAUDE.md
                // rule 3), so an old champion needs a paired A/B, not faith.
                "--retina-beings" => {
                    RETINA_BEINGS_ON.store(true, Ordering::Relaxed);
                }
                // Perceptual horizon, metres. A value <= 0 (or unparseable) keeps the
                // default rather than blinding the run: `visao::pitch_at` would clamp a
                // zero to 1 m anyway, and a silent 1 m retina is not a control arm.
                "--retina-road-view" => {
                    i += 1;
                    let value = args
                        .get(i)
                        .and_then(|v| v.parse::<f32>().ok())
                        .filter(|v| v.is_finite() && (20.0..=300.0).contains(v))
                        .unwrap_or_else(|| {
                            eprintln!("--retina-road-view requires metres in [20, 300]");
                            std::process::exit(2);
                        });
                    road_view_range_m = Some(value);
                }
                "--retina-range" => {
                    i += 1;
                    if i < args.len() {
                        let v: f32 = args[i].parse().unwrap_or(universe::visao::ALCANCE_PADRAO);
                        if v > 0.0 {
                            retina_range_m = v;
                        } else {
                            eprintln!(
                                "--retina-range {v} is not positive; keeping the default \
                                 {:.1} m",
                                universe::visao::ALCANCE_PADRAO
                            );
                        }
                    }
                }
                // Remove os buracos intencionais da pista (world.rs:141). Existia so no
                // caminho Bevy (main.rs:50); exposto aqui porque a bateria pos-refatoracao
                // mostrou a POPULACAO estacionada em 1.076 m por 20 geracoes — 11 m antes
                // do campo de buracos (1.087-1.092 m). Este A/B e o que separa "nao
                // aprendem a desviar" de "os buracos sao a segunda parede".
                "--no-holes" => {
                    universe::world::NO_HOLES.store(true, Ordering::Relaxed);
                }
                // Câmbio: travado (histórico, teto 21,5 m/s), auto (RPM, paridade Bevy)
                // ou neural (auto + intent[3]/[4]). Ver CAMBIO_MODO em statics.rs.
                "--gearbox-curriculum" => {
                    CAMBIO_CURRICULO.store(true, Ordering::Relaxed);
                }
                "--gearbox-ceiling" => {
                    i += 1;
                    if i < args.len() {
                        CAMBIO_TETO
                            .store(args[i].parse().unwrap_or(6).clamp(2, 6), Ordering::Relaxed);
                    }
                }
                "--gearbox" => {
                    i += 1;
                    if i < args.len() {
                        let m = match args[i].as_str() {
                            "travado" => 0u32,
                            "auto" => 1,
                            "neural" => 2,
                            other => {
                                // check-language: allow -- the three MODE names are the
                                // machine contract of the battery scripts; only the
                                // sentence around them is translated here.
                                eprintln!(
                                    "--gearbox '{other}' is invalid (travado|auto|neural); using travado"
                                );
                                0
                            }
                        };
                        CAMBIO_MODO.store(m, Ordering::Relaxed);
                    }
                }
                // Congela a mutação das linhas 5..48 de L2 (saídas que nenhum pipeline
                // lê — "feature neurons" sem consumidor; 4.128 pesos de busca inerte).
                "--l2-freeze-blind" => {
                    congela_l2_cegas = true;
                }
                "--no-l2-freeze-blind" => {
                    congela_l2_cegas = false;
                }
                // M3: regiões DIRIGIDAS (teto de referência) — ver evolution.rs::mascara_permite.
                "--region-mask" => {
                    mascara_regioes = true;
                }
                "--no-region-mask" => {
                    mascara_regioes = false;
                }
                // E-REG2 (M2): semeia τ por BLOCO (mesmos 4 blocos do M3) — reflexo
                // curto na parede/estrada, longo no associativo. Função segue tempo?
                "--tau-seeded" => {
                    tau_semeado = true;
                }
                "--no-tau-seeded" => {
                    tau_semeado = false;
                }
                // Visão periférica REAL: ch144-159 deixam de ser cópia bit-a-bit da
                // fóvea e passam a olhar as bordas da retina (ver perception.wgsl).
                // Off por padrão — muda a DISTRIBUIÇÃO de entrada, então campeão
                // antigo precisa de A/B pareado, não de fé.
                "--real-peripheral-vision" => {
                    universe::gpu::perception::PERIF_REAL.store(true, Ordering::Relaxed);
                }
                // M4: maestro de regiões (lento→gates com reserva). Ver statics.rs.
                "--neuromod-regions" => {
                    NEUROMOD_REGIOES.store(true, Ordering::Relaxed);
                }
                "--no-neuromod-regions" => {
                    NEUROMOD_REGIOES.store(false, Ordering::Relaxed);
                }
                // Laboratório de observação: grava a cadeia neural do ser 0 por tick
                // (394 f32/linha, ver context.rs). Usar com --retomar --gens 1.
                "--observe" => {
                    i += 1;
                    if i < args.len() {
                        observar_path = Some(args[i].clone());
                    }
                }
                // Open-loop screen, input side: records the raw 192-channel sensory
                // wave for being 0 (the same being --observe records), one row per
                // tick (195 f32/row, see gpu::wave_record). Off by default — forces a
                // per-tick readback of the whole wave buffer.
                "--record-wave" => {
                    i += 1;
                    if i < args.len() {
                        gravar_onda_path = Some(args[i].clone());
                    }
                }
                "--curve-portal" => {
                    portal_curva = true;
                }
                "--slow-steer" => {
                    i += 1;
                    if i < args.len() {
                        slow_steer_auth = args[i].parse().unwrap_or(0.20);
                    }
                }
                "--refuel-laps" => {
                    i += 1;
                    if i < args.len() {
                        refuel_laps = args[i].parse().unwrap_or(1).max(1);
                    }
                }
                "--life-cap" => {
                    i += 1;
                    if i < args.len() {
                        life_cap_arg = args[i].parse().unwrap_or(0);
                    }
                }
                "--n-elite" => {
                    i += 1;
                    if i < args.len() {
                        evo_n_elite = args[i].parse().unwrap_or(3);
                    }
                }
                "--tourney" => {
                    i += 1;
                    if i < args.len() {
                        evo_tourney = args[i].parse().unwrap_or(3);
                    }
                }
                "--gen-offset" => {
                    i += 1;
                    if i < args.len() {
                        gen_offset = args[i].parse().unwrap_or(0);
                    }
                }
                "--activation" => {
                    ativacao_on = true;
                }
                "--activation-n" => {
                    i += 1;
                    if i < args.len() {
                        ativacao_n = args[i].parse().unwrap_or(64).max(1);
                    }
                }
                "--regions" => {
                    regioes_on = true;
                }
                "--panel" => {
                    painel_on = true;
                }
                "--panel-port" => {
                    i += 1;
                    if i < args.len() {
                        painel_porta = args[i].parse().unwrap_or(3200);
                        painel_on = true;
                    }
                }
                "--tracks" => {
                    i += 1;
                    if i < args.len() {
                        pistas = parse_pistas(&args[i]);
                    }
                }
                // `--world <spec>` (repeatable): a parametric WorldSpec, see
                // `universe::world_spec::parse`. A bad spec exits loudly (2) instead of
                // silently falling back — unlike `--tracks`, this is a deliberate
                // difficulty experiment, so a typo should not quietly train on 8005.
                "--world" => {
                    i += 1;
                    if i < args.len() {
                        match universe::world_spec::parse(&args[i]) {
                            Ok(spec) => mundos.push(spec),
                            Err(e) => {
                                eprintln!("{e}");
                                std::process::exit(2);
                            }
                        }
                    }
                }
                "--world-file" => {
                    i += 1;
                    if i < args.len() {
                        match universe::world_spec::parse_world_file(std::path::Path::new(&args[i]))
                        {
                            Ok(mut specs) => mundos.append(&mut specs),
                            Err(e) => {
                                eprintln!("{e}");
                                std::process::exit(2);
                            }
                        }
                    }
                }
                "--track-held-out" => {
                    i += 1;
                    if i < args.len() {
                        pista_held_out = args[i].parse().ok();
                    }
                }
                "--held-out-every" => {
                    i += 1;
                    if i < args.len() {
                        held_out_cada = args[i].parse().unwrap_or(5).max(1);
                    }
                }
                "--replay-every" => {
                    i += 1;
                    if i < args.len() {
                        replay_every = args[i].parse().unwrap_or(1).max(1);
                    }
                }
                // ── FLY MAP (docs/FLY-MAP-BRAIN.md §5) ───────────────────────────
                // `--fly-map` is the switch that turns the whole subsystem on: without a
                // path there is no map, so `--fly-brain` and the four rates have nothing
                // to act on and the run takes exactly the old code path.
                "--fly-map" => {
                    i += 1;
                    if i < args.len() {
                        fly_map_path = Some(args[i].clone());
                    }
                }
                // Turns the map into the DRIVER (its three commands replace the fast
                // brain's intents) and, with it, the mutation of the `fly` block — the
                // same coupling `--hippocampus-sector` has, and for the same reason: that
                // block is the last consumer of `rng`, so a run that does not ask for it
                // must not draw from it at all.
                "--fly-brain" => {
                    FLY_BRAIN.store(true, Ordering::Relaxed);
                    universe::gpu::evolution::FLY_MUTATE_ON.store(true, Ordering::Relaxed);
                }
                "--fly-spiking" => {
                    fly_spiking_on = true;
                    FLY_SPIKING.store(true, Ordering::Relaxed);
                    // Spiking is itself a command source. Enable the same fly genome
                    // mutation path as --fly-brain so the selected connectome bindings
                    // can evolve during the run.
                    universe::gpu::evolution::FLY_MUTATE_ON.store(true, Ordering::Relaxed);
                }
                "--brain-control" => {
                    i += 1;
                    match args.get(i).map(String::as_str) {
                        Some("direct") => brain_control_direct = true,
                        Some("legacy") => brain_control_direct = false,
                        _ => {
                            eprintln!("[ERROR] --brain-control requires `direct` or `legacy`.");
                            std::process::exit(2);
                        }
                    }
                }
                "--fly-tau" => {
                    i += 1;
                    if i < args.len() {
                        fly_tau_steps = args[i].parse().unwrap_or(fly_tau_steps).max(1);
                    }
                }
                "--fly-p-bind" => {
                    i += 1;
                    if i < args.len() {
                        evo_fly_p_bind = args[i].parse().unwrap_or(evo_fly_p_bind);
                    }
                }
                "--fly-p-off" => {
                    i += 1;
                    if i < args.len() {
                        evo_fly_p_off = args[i].parse().unwrap_or(evo_fly_p_off);
                    }
                }
                "--fly-p-on" => {
                    i += 1;
                    if i < args.len() {
                        evo_fly_p_on = args[i].parse().unwrap_or(evo_fly_p_on);
                    }
                }
                "--fly-sigma" => {
                    i += 1;
                    if i < args.len() {
                        evo_fly_sigma = args[i].parse().unwrap_or(evo_fly_sigma);
                    }
                }
                "--fly-log-every" => {
                    i += 1;
                    if i < args.len() {
                        fly_log_every = args[i].parse().unwrap_or(0);
                    }
                }
                "--fly-spiking-trace" => {
                    i += 1;
                    if i < args.len() {
                        fly_spiking_trace_path = Some(args[i].clone());
                    }
                }
                "--fly-spiking-trace-every" => {
                    i += 1;
                    if i < args.len() {
                        fly_spiking_trace_every = args[i].parse().unwrap_or(64).max(1);
                    }
                }
                "--fly-spiking-trace-being" => {
                    i += 1;
                    if i < args.len() {
                        fly_spiking_trace_being = args[i].parse().unwrap_or(0);
                    }
                }
                // Anatomical binding (docs/FLY-MAP-BRAIN.md "Eye binding"): every neuron
                // the exporter wired to a retina pixel is driven by that pixel instead of
                // by the genome's channel binding. Requires a v2 map with the
                // pixel-binding block — checked below, once the map itself has loaded,
                // the same two-step shape as `--fly-brain` needing `--fly-map`.
                "--fly-eye" => {
                    universe::fly_map::FLY_EYE_ON.store(true, Ordering::Relaxed);
                }
                // Reads the anatomical steering reflex straight off the map instead of
                // evolving a readout (docs/FLY-MAP-BRAIN.md "Reflex readout"): checked
                // below, once the map itself has loaded, the same two-step shape as
                // `--fly-eye`/`--fly-brain` needing `--fly-map`.
                "--fly-reflex-alpha" => {
                    i += 1;
                    fly_reflex_alpha_v = brain_backend::finite_parameter(
                        args.get(i),
                        "--fly-reflex-alpha",
                        0.0,
                        1.0,
                    )
                    .unwrap_or_else(|error| {
                        eprintln!("[ERROR] {error}");
                        std::process::exit(2)
                    });
                }
                "--fly-reflex-gain" => {
                    i += 1;
                    fly_reflex_gain_v = brain_backend::finite_parameter(
                        args.get(i),
                        "--fly-reflex-gain",
                        0.0,
                        f32::MAX,
                    )
                    .unwrap_or_else(|error| {
                        eprintln!("[ERROR] {error}");
                        std::process::exit(2)
                    });
                }
                "--fly-readout-norm" => {
                    // The measured quantity (191v/193v) is the normalised asymmetry index, not
                    // the raw group difference; F3v showed the difference pins the wheel to one
                    // side. Off by default so every historical run stays bit-identical.
                    universe::fly_map::FLY_READOUT_NORM_ON.store(true, Ordering::Relaxed);
                }
                "--fly-readout" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!(
                            "[ERROR] --fly-readout requires a value: genome | reflex.\n\
                             [ERROR] `genome` is the default (the fly block's own \
                             out_pair genes select the readout, as today)."
                        );
                        std::process::exit(1);
                    }
                    match args[i].as_str() {
                        "genome" => {
                            universe::fly_map::FLY_READOUT_REFLEX_ON
                                .store(false, Ordering::Relaxed);
                        }
                        "reflex" => {
                            universe::fly_map::FLY_READOUT_REFLEX_ON.store(true, Ordering::Relaxed);
                        }
                        other => {
                            eprintln!(
                                "[ERROR] --fly-readout {other}: unknown mode.\n\
                                 [ERROR] Valid values are: genome | reflex.\n\
                                 [ERROR] Refusing to fall back to `genome` — a run that \
                                 silently swaps\n\
                                 [ERROR] the readout arm under test reports plausible \
                                 numbers for a\n\
                                 [ERROR] command source nobody asked for."
                            );
                            std::process::exit(1);
                        }
                    }
                }
                "--fly-cruise-throttle" => {
                    i += 1;
                    fly_cruise_throttle_v = brain_backend::finite_parameter(
                        args.get(i),
                        "--fly-cruise-throttle",
                        0.0,
                        1.0,
                    )
                    .unwrap_or_else(|error| {
                        eprintln!("[ERROR] {error}");
                        std::process::exit(2)
                    });
                }
                "--fly-reflex-sign" => {
                    i += 1;
                    if i < args.len() {
                        fly_reflex_sign_v = match args[i].as_str() {
                            "1" => 1.0,
                            "-1" => -1.0,
                            other => {
                                eprintln!(
                                    "[ERROR] --fly-reflex-sign {other}: only 1 or -1 are \
                                     valid — the sign convention that maps \"motion on \
                                     the left\" to a left or right turn is not \
                                     documented anywhere in this repository \
                                     (docs/FLY-MAP-BRAIN.md \"Reflex readout\"), so it \
                                     is an experiment arm, not a free gain."
                                );
                                std::process::exit(1);
                            }
                        };
                    }
                }
                "--resume" => {
                    retomar = true;
                }
                "--out" => {
                    i += 1;
                    if i < args.len() {
                        out_dir = args[i].clone();
                    }
                }
                "--help" => {
                    for flag in cli_flags::FLAGS {
                        println!(
                            "{}{}  {}",
                            flag.name,
                            if flag.takes_value { " VALUE" } else { "" },
                            flag.help
                        );
                    }
                    return;
                }
                _ => {
                    eprintln!("ERROR unknown argument {raw}; use --help for supported flags");
                    std::process::exit(2);
                }
            }
            i += 1;
        }
    }

    if start_formation && largada_aleatoria {
        eprintln!(
            "[ERROR] --start-formation uses a fixed grid and cannot be combined with --largada-aleatoria."
        );
        std::process::exit(2);
    }

    let ablation_plan = match (ablate_spec.as_deref(), ablate_from_tick) {
        (None, None) => None,
        (Some(_), None) => {
            eprintln!("ERROR --ablate requires --ablate-from-tick");
            std::process::exit(2)
        }
        (None, Some(_)) => {
            eprintln!("ERROR --ablate-from-tick requires --ablate");
            std::process::exit(2)
        }
        (Some(spec), Some(from_tick)) => Some(
            universe::gpu::brain::AblationPlan::parse(spec, ablate_being, from_tick, n_seres)
                .unwrap_or_else(|error| {
                    eprintln!("ERROR --ablate: {error}");
                    std::process::exit(2)
                }),
        ),
    };
    if brain_control_direct && ablation_plan.is_some() {
        eprintln!(
            "[ERROR] --ablate targets the native fast brain and cannot be combined with direct fly brain control."
        );
        std::process::exit(2);
    }

    if !matches!(
        modo_fitness.as_str(),
        "dist" | "prog" | "checkpoint" | "centro" | "checkpoint-legacy" | "centro-legacy"
    ) {
        eprintln!(
            "[ERROR] --fitness '{}' is invalid; expected dist, prog, checkpoint, centro, checkpoint-legacy, or centro-legacy",
            modo_fitness
        );
        std::process::exit(2);
    }
    FUEL_AUTO_K.store(fuel_auto_k.unwrap_or(0.0), Ordering::Relaxed);

    // `--record-wave` writes the UWAV header once, before the generation loop
    // (`ctx.wave_writer` setup, below) — but the loop itself resets CfC state and runs a
    // fresh JIT warm-up tick every generation (`ctx.reset_cfc` and the warm-up `ctx.tick`
    // call further down). With `--gens > 1` that makes an N-generation recording N
    // independent episodes concatenated with no boundary marker: `WaveConfig` has no
    // `gens` field, and `screen.rs` treats only row 0 as warm-up and replays everything
    // after it as one continuous episode. Every row past generation 1 would then be
    // scored against a `ScreenBrain` whose state never reset — wrong numbers, no error.
    // Refusing here is cheaper than a format bump and does not touch the existing
    // calibration recording, which was always made with `--gens 1`.
    if gravar_onda_path.is_some() && n_gens > 1 {
        eprintln!(
            "ERROR --record-wave requires --gens 1: the recording format has no generation \
             boundary marker, so a multi-generation run would concatenate N independent \
             episodes into one file that screen.rs would replay as a single continuous \
             episode (CfC/period-counter state never resets across generations in the \
             recording, but it does reset in every generation of the run that produced it). \
             Pass --gens 1, or record each generation into its own file."
        );
        std::process::exit(2);
    }

    // ── FLY MAP (docs/FLY-MAP-BRAIN.md §1) ────────────────────────────────────
    //
    // Loaded HERE, before the first genome exists: the `fly` block is `200 + n_types`
    // floats long and `Genome::random` reads that shape from the layout this publishes.
    // Without `--fly-map` nothing in this block runs, no genome carries the block, and no
    // buffer is created.
    //
    // A map that was asked for and did not load is a hard error, never a fallback: the
    // run would otherwise report plausible numbers for a brain nobody asked for — the
    // same rule `--resume` already follows two hundred lines below.
    let fly_map: Option<universe::fly_map::FlyMap> = match fly_map_path.as_deref() {
        None => None,
        Some(path) => match universe::fly_map::FlyMap::load(path) {
            Ok(m) => {
                println!(
                    "  [fly-map] n={} m={} k_in={} k_out={} n_types={} · {:.1} MB · {}",
                    m.n(),
                    m.m(),
                    m.k_in,
                    m.k_out(),
                    m.n_types,
                    m.resident_bytes() as f64 / (1024.0 * 1024.0),
                    path
                );
                universe::fly_map::set_layout(m.fly_layout());
                Some(m)
            }
            Err(e) => {
                eprintln!("[ERROR] --fly-map {path}: {e}");
                std::process::exit(1);
            }
        },
    };
    if fly_map.is_none() && (fly_brain() || fly_spiking()) {
        eprintln!(
            "[ERROR] --fly-brain/--fly-spiking without --fly-map: there is no map to read commands from.\n\
             [ERROR] Pass --fly-map <flymap v2 file>, or drop the fly driver flag."
        );
        std::process::exit(1);
    }
    if brain_control_direct && !(fly_brain() || fly_spiking()) {
        eprintln!("[ERROR] --brain-control requires --fly-brain or --fly-spiking.");
        std::process::exit(2);
    }
    if brain_control_direct
        && (piloto != 0.0
            || piloto_thr != 0.0
            || piloto_sensor != 0.0
            || piloto_formula_path.is_some()
            || piloto_forja_path.is_some()
            || ativacao_on
            || observar_path.is_some()
            || brain_skip != 1
            || steer_fixo != 0.0
            || steer_escala != 0.0
            || thr_fixo != 0.0
            || hippocampus_on())
    {
        eprintln!(
            "[ERROR] direct brain control cannot combine pilots, fixed motor overrides, hippocampus, native observation/activation, or native brain skipping."
        );
        std::process::exit(2);
    }
    if fly_brain() && fly_spiking() {
        eprintln!(
            "[ERROR] --fly-brain and --fly-spiking select different fly numerical models; choose one."
        );
        std::process::exit(1);
    }
    if fly_spiking_on
        && fly_map
            .as_ref()
            .map(|m| m.pre_csr.is_none())
            .unwrap_or(false)
    {
        eprintln!(
            "[ERROR] --fly-spiking requires a flymap v2 file with the presynaptic CSR (flags bit 0)."
        );
        std::process::exit(1);
    }
    if fly_spiking_trace_path.is_some() && !fly_spiking_on {
        eprintln!(
            "[ERROR] --fly-spiking-trace requires --fly-spiking: the trace reads the LIF state, not the rate model."
        );
        std::process::exit(1);
    }
    if fly_spiking_trace_being >= n_seres {
        eprintln!(
            "[ERROR] --fly-spiking-trace-being={} is outside the population of {} beings.",
            fly_spiking_trace_being, n_seres
        );
        std::process::exit(1);
    }
    // --fly-eye requires a v2 map that carries the pixel-binding block (docs/FLY-MAP-
    // BRAIN.md "Eye binding") — checked here, not inside `FlyMapGpu`, so a run that asked
    // for the eye and cannot have it fails before a single tick runs rather than silently
    // reporting a fast brain nobody asked for (same rule as `--fly-brain`/`--resume`
    // above and below).
    if universe::fly_map::eye_on() {
        match &fly_map {
            None => {
                eprintln!(
                    "[ERROR] --fly-eye without --fly-map: there is no map to bind pixels \
                     on.\n\
                     [ERROR] Pass --fly-map <flymap v2 file with the pixel-binding block>, \
                     or drop --fly-eye."
                );
                std::process::exit(1);
            }
            Some(m) if !m.has_pixel_binding() => {
                eprintln!(
                    "[ERROR] --fly-eye requires a flymap v2 file with the pixel-binding \
                     block (flags bit 1).\n\
                     [ERROR] The loaded map carries none — re-export with the Python \
                     side's pixel-binding pass, or drop --fly-eye."
                );
                std::process::exit(1);
            }
            Some(_) => {}
        }
    }
    // --fly-readout reflex requires a v3 map carrying the reflex-readout block
    // (docs/FLY-MAP-BRAIN.md "Reflex readout") — checked here, before a single tick
    // runs, mirroring the --fly-eye check above and the --fly-brain check further up.
    // `FLY_REFLEX_THROTTLE_IS_NONE` is decided HERE, once, from the CPU-side `FlyMap`
    // (`context.rs::tick()` only ever sees `FlyMapGpu`, which does not carry
    // `reflex_out`) — the same "publish the fact once, at load, through a static"
    // pattern `tau_bank::MODE` and `FLY_BRAIN` already use.
    if universe::fly_map::readout_reflex_on() {
        if let Err(msg) = universe::fly_map::readout_reflex_requires_the_block(fly_map.as_ref()) {
            eprintln!("[ERROR] {msg}\n[ERROR] Pass a v3 map, or drop --fly-readout reflex.");
            std::process::exit(1);
        }
        let reflex_out = fly_map
            .as_ref()
            .and_then(|m| m.reflex_out)
            .expect("readout_reflex_requires_the_block already confirmed the block");
        FLY_REFLEX_THROTTLE_IS_NONE.store(
            universe::fly_map::reflex_throttle_is_none(reflex_out),
            Ordering::Relaxed,
        );
    }
    FLY_CRUISE_THROTTLE.store(fly_cruise_throttle_v, Ordering::Relaxed);
    FLY_REFLEX_SIGN.store(fly_reflex_sign_v, Ordering::Relaxed);
    FLY_REFLEX_ALPHA.store(fly_reflex_alpha_v, Ordering::Relaxed);
    FLY_REFLEX_GAIN.store(fly_reflex_gain_v, Ordering::Relaxed);

    let inicio = if retomar {
        "retomando de checkpoint"
    } else {
        "pesos ALEATÓRIOS (σ=0.30, N(0,σ))"
    };
    let spectral_init_on = universe::gpu::evolution::SPECTRAL_INIT_ON.load(Ordering::Relaxed);
    println!(
        "=== gpu_evolve: {} seres · {} ticks/gen · {} gens ===",
        n_seres, n_ticks, n_gens
    );
    println!("  Inicio: {}", inicio);
    println!(
        "  Init espectral L1 (P0.5): {}",
        if spectral_init_on {
            "LIGADA — L1 nasce com ganho espectral ≈ 1.0 (alvo em thresholds::SPECTRAL_INIT_TARGET)"
        } else {
            "DESLIGADA (--sem-init-espectral) — L1 nasce cru, σ=0.30 i.i.d. sem correção (braço de controle)"
        }
    );
    let driving_reflex_birth =
        universe::gpu::evolution::DRIVING_REFLEX_BIRTH_ON.load(Ordering::Relaxed);
    println!(
        "  Motor birth wiring: {}",
        if driving_reflex_birth {
            "sparse evolvable reflexes (alignment/curve risk/clearance -> 3 motor pathways)"
        } else {
            "fully random motor pathways (control arm)"
        }
    );
    if taxi_reference_witness && (!organism_taxi || n_gens != 1 || n_seres != 1 || ecology_on || resume_state.is_some() || snapshot_every > 0 || !driving_diagnostics_on) {
        eprintln!("--taxi-reference-witness requires organism-taxi-v1, one being, one generation, --driving-diagnostics, no ecology/resume/snapshots; never an evolution arm");
        std::process::exit(2);
    }
    if school_lesson.is_some() && (!organism_taxi || ecology_on || resume_state.is_some()) {
        eprintln!("--school requires --scenario organism-taxi-v1 and no ecology/resume");
        std::process::exit(2);
    }
    if school_curriculum.is_some() {
        if school_lessons.is_empty() {
            school_lessons = match school_lesson {
                Some(lesson) => vec![lesson],
                None => universe::gpu::school::Lesson::ALL.to_vec(),
            };
        }
        if school_lesson.is_none() {
            school_lesson = Some(school_lessons[0]);
        }
    } else if !school_lessons.is_empty() {
        eprintln!("--school-lessons requires --school-curriculum adaptive|uniform");
        std::process::exit(2);
    }
    if school_curriculum.is_some() && (!organism_taxi || ecology_on || resume_state.is_some()) {
        eprintln!("--school-curriculum requires --scenario organism-taxi-v1 and no ecology/resume");
        std::process::exit(2);
    }
    if school_start_fold != universe::organism_school::LessonFold::All && school_lesson.is_none() {
        eprintln!("--school-start-fold requires --school or --school-curriculum");
        std::process::exit(2);
    }
    if school_stop_shaping && school_lesson.is_none() {
        eprintln!("--school-stop-shaping requires --school or --school-curriculum");
        std::process::exit(2);
    }
    if school_shared_start.is_some() && school_lesson.is_none() {
        eprintln!("--school-shared-start requires --school or --school-curriculum");
        std::process::exit(2);
    }
    if school_eval_starts > 1 && school_lesson.is_none() {
        eprintln!("--school-eval-starts greater than 1 requires --school or --school-curriculum");
        std::process::exit(2);
    }
    if school_eval_starts > 1 && school_shared_start.is_some() {
        eprintln!("--school-eval-starts greater than 1 conflicts with --school-shared-start; a batch must contain distinct starts");
        std::process::exit(2);
    }
    if school_eval_starts > 1 && !school_bench {
        eprintln!("--school-eval-starts greater than 1 requires --school-mode bench so the aggregate has one declared objective");
        std::process::exit(2);
    }
    if let Some(freeze_at) = plasticity_freeze_after {
        if lifetime_flags & 1 == 0 {
            eprintln!("--plasticity-freeze-after requires --plasticity");
            std::process::exit(2);
        }
        if freeze_at >= n_ticks {
            eprintln!("--plasticity-freeze-after {freeze_at} must be below --ticks {n_ticks}");
            std::process::exit(2);
        }
    }
    let school_contract = universe::gpu::school::LessonContract::default();
    let school_horizon_ticks = school_horizon.unwrap_or(n_ticks as u32);
    if school_lesson.is_some() && school_horizon_ticks as usize > n_ticks {
        eprintln!("--school-horizon {} exceeds --ticks {n_ticks}", school_horizon_ticks);
        std::process::exit(2);
    }
    let genome_count = n_seres;
    if school_eval_starts > 1 {
        n_seres = genome_count.checked_mul(school_eval_starts).unwrap_or_else(|| {
            eprintln!("school replica population overflows usize");
            std::process::exit(2);
        });
        println!(
            "  [school-batch] {genome_count} genomes × {school_eval_starts} starts = {n_seres} GPU slots"
        );
    }
    if organism_taxi && !args.iter().any(|a| a == "--ecology-min-progress") { ecology_progress = 1.0; }
    if organism_taxi && ecology_on && n_gens != 1 {
        eprintln!("ERROR continuous organism ecology requires --gens 1; use --ticks for the lifetime horizon");
        std::process::exit(2);
    }
    if (snapshot_every > 0 || resume_state.is_some()) && !organism_taxi {
        eprintln!("runtime checkpoints require --scenario organism-taxi-v1");
        std::process::exit(2);
    }
    if evolving_body && !fuel_on && !organism_taxi {
        eprintln!("ERROR --evolving-body requires --fuel to express engine resource costs");
        std::process::exit(2);
    }
    println!(
        "  Body evolution: {} (capacity_mass_tradeoff_v1)",
        if evolving_body {
            "enabled; inherited engine/brake capacity"
        } else {
            "fixed historical body"
        }
    );
    let sensors_evolve_on = universe::gpu::evolution::SENSORS_EVOLVE_ON.load(Ordering::Relaxed);
    println!(
        "  Sensores (FASE SENSORES, degrau 1): {}",
        if sensors_evolve_on {
            "EVOLUINDO (--sensores-evolutivos) — FOV/alcance-parede/alcance-lateral/curvatura heterogêneos e mutáveis"
        } else {
            "FIXOS (padrão / --sensores-fixos) — todo genoma em SENSOR_DEFAULTS, os hardcodes de hoje"
        }
    );
    // P1.7 — cérebro assíncrono: relatório de quais dos três mecanismos estão
    // mutando nesta corrida. Genoma sempre nasce neutro (gen 0 idêntica ao
    // cérebro de hoje); o que muda entre "ligado"/"--sem-*" é se a mutação afasta
    // o campo do neutro ao longo das gerações.
    println!("  Cérebro assíncrono (P1.7):");
    println!(
        "    memória por neurônio:  {}",
        if sem_memoria_neuronal {
            "CONGELADA (--sem-memoria-neuronal) — alfa preso em 0.0, braço de controle"
        } else {
            "mutando (padrão) — h[n]=h[n-1]·α+x[n]·(1-α), α∈[0,0.95] a partir de 0.0"
        }
    );
    println!(
        "    período/limiar (L1):   {}{}",
        if sem_periodo_neuronal {
            "período CONGELADO (--sem-periodo-neuronal) "
        } else {
            "período mutando (padrão) "
        },
        if sem_limiar_neuronal {
            "· limiar CONGELADO (--no-neuron-threshold)"
        } else {
            "· limiar mutando (padrão)"
        }
    );
    println!(
        "    ativação heterogênea:  {}",
        if sem_ativacao_heterogenea {
            "CONGELADA (--sem-ativacao-heterogenea) — todos em tanh, braço de controle"
        } else {
            "mutando (padrão) — tanh/ReLU/linear/leaky-ReLU, taxa 0,2%/neurônio/geração"
        }
    );
    let motor_p_mutate = evo_motor_p_mutate.unwrap_or(evo_p_mutate);
    let motor_sigma = evo_motor_sigma.unwrap_or(evo_sigma);
    if !motor_p_mutate.is_finite()
        || !(0.0..=1.0).contains(&motor_p_mutate)
        || !motor_sigma.is_finite()
        || motor_sigma < 0.0
    {
        eprintln!("[ERROR] motor mutation requires p in [0,1] and finite sigma >= 0");
        std::process::exit(1);
    }
    println!(
        "  Evolution: sigma={:.3}  p_mutate={:.3}  motor_sigma={:.3}  motor_p={:.3}  n_elite={}  tourney={}  hip_p={:.2}  hip_sigma={:.2}",
        evo_sigma,
        evo_p_mutate,
        motor_sigma,
        motor_p_mutate,
        evo_n_elite,
        evo_tourney,
        evo_hip_p,
        evo_hip_sigma
    );
    println!(
        "  [tau] bank: --tau-bank {} — {}",
        tau_bank_mode().as_flag(),
        tau_bank_mode().describe()
    );
    println!("  SEM muletas motoras (escape_ticks, curvature_governor)");
    println!(
        "  Fitness: {} · wall_cost={:.3} m/s · impact_cost={:.3} m/kJ",
        modo_fitness, wall_cost, impact_cost
    );
    println!();

    // ── Rotação de pistas (P1, docs/PLANO_EVOLUCAO.md) ────────────────────────
    // Aviso não-fatal: se a pista reservada também estiver no conjunto de treino,
    // ela deixa de medir generalização — o campeão veria a "reservada" durante o
    // treino normal, então nunca é surpreendido por ela no bloco [generalizacao].
    if let Some(ho) = pista_held_out {
        if pistas.contains(&ho) {
            eprintln!(
                "[AVISO] --pista-held-out {} também está em --pistas — ela não é mais \
                 reservada, a medição de generalização fica inválida.",
                ho
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
            if !seeds_a_checar.contains(&ho) {
                seeds_a_checar.push(ho);
            }
        }
        let mut alguma_falhou = false;
        for &s in &seeds_a_checar {
            let w = {
                let _q = universe::world::silence_map_diag();
                VoxelWorld::new(s, track_width)
            };
            let ate = w.lap_distance.max(1.0);
            let defeitos = w.validar_pista(track_width, ate);
            if !defeitos.is_empty() {
                alguma_falhou = true;
                eprintln!(
                    "[pista_check] seed {s}: {} defeito(s) em {ate:.0}m de eixo — {}{}",
                    defeitos.len(),
                    defeitos
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" | "),
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
    // Índice do currículo: anda por MÉRITO (volta completada), não por geração.
    let mut idx_curriculo: usize = gen_offset;
    let (mut world, mut world_seed_atual, mut world_key_atual) = if organism_taxi {
        let _quiet = universe::world::silence_map_diag();
        resolve_world(&pistas, &mundos, gen_offset, track_width)
    } else {
        resolve_world(&pistas, &mundos, gen_offset, track_width)
    };
    let mut organism_city = organism_taxi.then(|| universe::organism_city::build_city(&mut world));
    let mut organism_stations = if organism_taxi {
        let stations = organism_city.as_ref().unwrap().stations.clone();
        universe::organism_scenario::validate_city(&world, &stations).expect("invalid taxi city");
        stations
    } else {
        Vec::new()
    };
    let mut lap_dist = world.lap_distance as f64;
    if organism_taxi {
        println!(
            "city: {}×{} cells · {} road-axis waypoints · {} stations [{:.0}ms] · seed={}",
            world.size, world.size, world.waypoints_world.len(), organism_stations.len(),
            t0.elapsed().as_secs_f64() * 1000.0, world_seed_atual
        );
    } else {
        println!(
            "mundo: {}×{} · {} wp · volta={:.0}m  [{:.0}ms] · pista inicial seed={}",
            world.size, world.size, world.waypoints_world.len(), lap_dist,
            t0.elapsed().as_secs_f64() * 1000.0, world_seed_atual
        );
    }
    // ── Perceptual horizon (`--retina-range`) ─────────────────────────────────
    // Set BEFORE `GpuCtx::new`, which bakes `visao::pitch` into the retina uniform.
    // `visao::definir` is a `OnceLock::set`, so this is the one and only call in the
    // process; with the flag absent the value is `ALCANCE_PADRAO` and every consumer
    // takes its identity branch (`pitch_at` returns the const table, `range_scale_of`
    // returns exactly 1.0). Printed unconditionally, next to MAP DIAG: a run whose
    // horizon you cannot read off the log is an unlabelled experiment. The four sensor
    // numbers on the line are the BIRTH values (`SENSOR_DEFAULTS`) after scaling; under
    // `--evolving-sensors` each being carries its own, scaled by the same factor.
    if let Some(range) = road_view_range_m {
        universe::visao::set_road_view(range).expect("validated road-view range");
        println!(
            "   [retina-road-view] physical-v1; log ground sampling 4..{range} m; independent probes unchanged"
        );
    }
    universe::visao::definir(universe::visao::Params {
        alcance_m: retina_range_m,
        fov_deg: universe::config::iara::vision::FOV_DEG,
    });
    {
        let s = universe::visao::range_scale();
        let d = universe::gpu::evolution::SENSOR_DEFAULTS;
        let ground_horizon =
            universe::visao::ROAD_VIEW_REFERENCE_EYE_HEIGHT_M / -universe::visao::pitch(0).tan();
        use universe::gpu::evolution::sensor_idx as sx;
        println!(
            "   [retina] nominal ground horizon = {:.1} m (probe scale x{:.3}) · ideal braking bound {:.1} m/s · wall {:.1} m · \
             lateral {:.1} m · curvature {:.0}/{:.0}/{:.0} m",
            ground_horizon,
            s,
            universe::visao::velocidade_cega(ground_horizon, 11.4),
            d[sx::WALL_RANGE_M] * s,
            d[sx::LAT_RANGE_M] * s,
            d[sx::CURV_NEAR_M] * s,
            d[sx::CURV_MID_M] * s,
            d[sx::CURV_FAR_M] * s,
        );
    }
    if !pistas.is_empty() {
        println!(
            "  [pistas] rotação LIGADA: {:?} (gen % {})",
            pistas,
            pistas.len()
        );
    }
    if let Some(ho) = pista_held_out {
        println!(
            "  [pistas] held-out seed={} a cada {} gerações (medição pura, NUNCA entra na seleção)",
            ho, held_out_cada
        );
    }
    println!();

    // ── Painel HTTP tempo-real (E5, `--panel`) ────────────────────────────────
    // Lançado ANTES do GPU subir de propósito: `GpuCtx::new` pode levar segundos, e o
    // traçado (`set_track`) já fica disponível em `/api/status` enquanto isso acontece.
    let painel: Option<painel::PainelHandle> = if painel_on {
        let p = painel::PainelHandle::launch(painel_porta);
        p.set_track(&world.waypoints_world);
        Some(p)
    } else {
        None
    };

    // Diagnostics writes these modes during construction, before the first tick.
    MOTOR_HEAD_MODE.store(motor_head_mode, Ordering::Relaxed);
    SLOW_CANONICAL_KINEMATICS.store(canonical_slow_kinematics, Ordering::Relaxed);
    CONTINUOUS_WALL_PROBES.store(continuous_probes, Ordering::Relaxed);
    THROTTLE_ARBITRATION_MODE.store(throttle_arbitration_mode, Ordering::Relaxed);
    REVERSE_CONTROLLER_V2.store(
        reverse_controller_mode == reverse_controller::ReverseMode::V2,
        Ordering::Relaxed,
    );

    // ── GPU ───────────────────────────────────────────────────────────────────
    let t1 = Instant::now();
    let mut ctx = match GpuCtx::new(&world, n_seres) {
        Some(c) => c,
        None => {
            eprintln!("ERRO: sem GPU utilizável. Abortando.");
            std::process::exit(1);
        }
    };
    if let Some(plan) = ablation_plan.as_ref() {
        ctx.brain.set_ablation(plan, n_seres);
    }
    ctx.erase_fast_memory = erase_fast_memory;
    ctx.feedback_mode = feedback_mode;
    ctx.collect_contact_metrics =
        impact_cost > 0.0 || DANO_ESCALA.load(Ordering::Relaxed) > 0.0 || driving_diagnostics_on;
    ctx.brain_skip = brain_skip;
    ctx.brain_control_direct = brain_control_direct;
    // FLY MAP: the connectome goes resident on the SAME device as every other kernel —
    // `GpuCtx` owns the device, so the map shares it instead of opening a second one
    // (`src/gpu/CLAUDE.md`, "one device, all buffers resident").
    if let Some(map) = fly_map.as_ref() {
        // The two fly implementations own separate state layouts. Keep the selected
        // model only: allocating the legacy rate buffers alongside the spiking ring
        // needlessly duplicates the connectome's hot state on large maps.
        if fly_spiking_on {
            match universe::gpu::fly_map::FlySpikingGpu::new(
                std::sync::Arc::clone(&ctx.device),
                std::sync::Arc::clone(&ctx.queue),
                map,
                n_seres,
                universe::fly_map::spiking::SpikingParams::default(),
            ) {
                Some(gpu) => {
                    println!(
                        "  [fly-spiking] delayed LIF + integer synapse accumulator resident on the GPU · {}",
                        if fly_spiking() {
                            "DRIVING the beings (--fly-spiking)"
                        } else {
                            "stepping only"
                        }
                    );
                    ctx.fly_spiking = Some(gpu);
                }
                None => {
                    eprintln!(
                        "[ERROR] --fly-spiking: the map lacks a valid presynaptic CSR or does not fit on this device."
                    );
                    std::process::exit(1);
                }
            }
        } else {
            match universe::gpu::fly_map::FlyMapGpu::new(
                std::sync::Arc::clone(&ctx.device),
                std::sync::Arc::clone(&ctx.queue),
                map,
                n_seres,
                fly_tau_steps,
            ) {
                Some(gpu) => {
                    println!(
                        "  [fly-map] resident on the GPU · alpha = 1/{} · {}",
                        fly_tau_steps,
                        if fly_brain() {
                            "DRIVING the beings (--fly-brain)"
                        } else {
                            "stepping only — the fast brain still drives"
                        }
                    );
                    ctx.fly = Some(gpu);
                }
                None => {
                    eprintln!(
                        "[ERROR] --fly-map: the map does not fit on this device (see above)."
                    );
                    std::process::exit(1);
                }
            }
        }
    }
    if let Some(ref p) = piloto_forja_path {
        match piloto_forja::PilotoForja::carregar(p) {
            Some(pf) => {
                println!(
                    "  [piloto-forja] ser 0 dirigido pela base {} ({} features, {} mult/tick)",
                    pf.base(),
                    pf.n_features(),
                    pf.mult_por_tick()
                );
                ctx.piloto_forja = Some(pf);
            }
            None => eprintln!("[ERRO] não consegui carregar o piloto v2 de {p}"),
        }
    }
    let selected_brain =
        brain_backend::BrainBackend::from_options(ctx.fly.as_ref(), ctx.fly_spiking.as_ref());
    let schema = universe::brain_interface::OBSERVATION_SCHEMA_V1;
    println!(
        "  [brain-interface] model={} mode={} retina={} wave={} motor_channels=3",
        selected_brain.name(),
        if brain_control_direct {
            "direct"
        } else {
            "legacy"
        },
        schema.retina_samples,
        schema.wave_channels
    );

    if let Some(ref p) = piloto_formula_path {
        match piloto_linear::PilotoLinear::carregar(p) {
            Some(pl) => {
                println!(
                    "  [piloto-formula] ser 0 dirigido pela fórmula ({} entradas)",
                    pl.n_entradas()
                );
                let mut pl = pl;
                pl.prob_aleatoria = portao_aleatorio;
                pl.semente = portao_semente;
                if portao_aleatorio > 0.0 {
                    println!(
                        "  [piloto] CONTROLE: portão ALEATÓRIO com p={portao_aleatorio:.3} (ignora o estado)"
                    );
                }
                ctx.piloto_formula = Some(pl);
            }
            None => {
                eprintln!("ERRO: --piloto-formula não carregou {p}");
                std::process::exit(1);
            }
        }
    }
    if ruido_sigma > 0.0 || ruido_drop > 0.0 {
        ctx.brain.set_noise(ruido_sigma, ruido_drop);
        println!("  [ruido] D3: sigma={ruido_sigma} dropout={ruido_drop}");
    }
    if clima_mu < 1.0 || vento_x != 0.0 || vento_z != 0.0 {
        ctx.phys_gpu
            .set_clima(clima_mu, vento_x, vento_z, clima_freq);
        println!("  [clima] D5: mu={clima_mu} freq={clima_freq} vento=({vento_x},{vento_z}) N");
    }
    // Estado da escada/fantasma (vivem no laço de gerações).
    let mut escada_ctl = if escada_on {
        Some(escada::Escada::padrao())
    } else {
        None
    };
    let mut fantasma_atual: fantasma::Fantasma = fantasma::Fantasma::default();
    let mut hist_med_escada: Vec<f64> = Vec::new();
    /// D8 — per-family reference use `ref_F`, FROZEN at the first generation the flag
    /// is on and never recomputed.
    ///
    /// Frozen because a reference that follows the population is not a tax, it is a
    /// ranking: if `ref_F` is always the current median, half the population always
    /// pays above reference no matter how much everyone prunes, and the pressure can
    /// never be satisfied — the incentive to prune disappears exactly when the
    /// population starts responding to it. A frozen reference means "cheaper than the
    /// population WAS" is a real, reachable win.
    let mut ref_familias: Vec<f32> = Vec::new();
    if CAMBIO_CURRICULO.load(Ordering::Relaxed) {
        // O valor de --cambio-teto vira o MÁXIMO; o teto vivo começa em 2 (≡ travado).
        CAMBIO_TETO_MAX.store(CAMBIO_TETO.load(Ordering::Relaxed), Ordering::Relaxed);
        CAMBIO_TETO.store(2, Ordering::Relaxed);
        println!(
            "  [cambio] currículo de teto LIGADO: começa na marcha 2, sobe por mérito até {}",
            CAMBIO_TETO_MAX.load(Ordering::Relaxed)
        );
    }
    if let Some(ref path) = observar_path {
        match std::fs::File::create(path) {
            Ok(f) => {
                ctx.obs_writer = Some(std::io::BufWriter::new(f));
                println!("  [observar] gravando cadeia neural do ser 0 em {path} (394 f32/tick)");
            }
            Err(e) => {
                eprintln!("ERRO --observe {path}: {e}");
                std::process::exit(1);
            }
        }
    }
    if let Some(ref path) = gravar_onda_path {
        match std::fs::File::create(path) {
            Ok(f) => {
                ctx.wave_writer = Some(std::io::BufWriter::new(f));
                println!(
                    "  [gravar-onda] recording being 0's raw sensory wave to {path} \
                     ({} f32/row)",
                    universe::gpu::wave_record::ROW_F32
                );
            }
            Err(e) => {
                eprintln!("ERROR --record-wave {path}: {e}");
                std::process::exit(1);
            }
        }
    }
    if let Some(ref path) = fly_spiking_trace_path {
        match ctx.enable_spiking_trace(path, fly_spiking_trace_every, fly_spiking_trace_being) {
            Ok(()) => println!(
                "  [fly-spiking-trace] recording being {} every {} ticks to {} (compact GPU snapshot)",
                fly_spiking_trace_being, fly_spiking_trace_every, path
            ),
            Err(e) => {
                eprintln!("ERROR --fly-spiking-trace {path}: {e}");
                std::process::exit(1);
            }
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
    // A pista INICIAL também precisa do tanque calculado — sem isto `--fuel-auto`
    // only applied after the first track change (see `automatic_fuel_tank`).
    {
        let k = FUEL_AUTO_K.load(Ordering::Relaxed);
        if k > 0.0 {
            let x = automatic_fuel_tank(world.lap_distance as f64, k, setor_paga);
            FUEL_TANK.store(x, Ordering::Relaxed);
            if !organism_taxi {
                println!(
                    "  [fuel-auto] pista inicial (volta {:.0} m) → tanque {x:.4} (k={k})",
                    world.lap_distance
                );
            }
        }
    }
    SETOR_PAGA.store(setor_paga, Ordering::Relaxed);
    // ── HOSTILE WORLD: defaults, refusals and the ceilings (BODY PHASE) ───────
    //
    // Runs AFTER `SETOR_PAGA` on purpose: D7 has to be able to take the sector
    // refill away, and it can only do that once the flag has been stored.
    //
    // Everything printed here is arithmetic done BEFORE any run (CLAUDE.md rule 4).
    // The point is not decoration: a hard 0% or a hard 100% is a metric hitting its
    // limit, and the only way to tell that apart from a result is to have written the
    // limit down first.
    {
        use universe::gpu::perception::BODY_CHANNELS_ON;

        // Defaults for the dose knobs. `AtomicF32::new()` is 0.0, and 0.0 means "off"
        // for the master switches (`--resource-scarce`, `--predator`, `--tire-thermal`,
        // `--sensor-cost`) but would mean "broken" for the shape knobs — a station
        // with radius 0 can never be reached, and that is a pressure that is silently
        // off, the exact failure the dose canary exists to catch.
        let default_f32 = |s: &AtomicF32, v: f32| {
            if s.load(Ordering::Relaxed) <= 0.0 {
                s.store(v, Ordering::Relaxed);
            }
        };
        default_f32(&STATION_CAPACITY, universe::resources::DEFAULT_CAPACITY);
        default_f32(
            &STATION_REGEN,
            universe::resources::DEFAULT_RECHARGE_PER_TICK,
        );
        default_f32(&STATION_RADIUS_M, universe::resources::DEFAULT_RADIUS_M);
        default_f32(
            &STATION_TRANSFER,
            universe::resources::DEFAULT_TRANSFER_PER_TICK,
        );
        default_f32(&PREDATOR_DRAIN, 4.0);
        default_f32(&PREDATOR_RADIUS_M, 12.0);
        default_f32(&PREDATOR_PERCEPT_M, 60.0);
        default_f32(&PREDATOR_SPAWN_BACK_M, 150.0);
        default_f32(&SENSOR_COST_CAP, 2.0);

        let n_stations = STATIONS_N.load(Ordering::Relaxed);
        let pred_frac = PREDATOR_SPEED_FRAC.load(Ordering::Relaxed);
        let thermal_gain = TIRE_THERMAL_GAIN.load(Ordering::Relaxed);
        let sensor_dose = SENSOR_COST.load(Ordering::Relaxed);
        let body_on = BODY_CHANNELS_ON.load(Ordering::Relaxed);

        if n_stations > 0 {
            // REFUSAL, not a warning. `--resource-scarce` without fuel consumption is
            // "decoration by construction" (D7 spec): the stations would hand out a
            // resource nothing spends, every dose would produce the same number, and
            // the run would look like a measurement of a pressure that never existed.
            if !FUEL_ON.load(Ordering::Relaxed) {
                eprintln!(
                    "[recurso] ERROR: --resource-scarce {n_stations} requires --fuel or \
                     --fuel-auto. With nothing consuming it, a station is decoration and every \
                     dose would produce the same number."
                );
                std::process::exit(2);
            }
            // D7 REPLACES the per-sector refill; it does not sit next to it. Leaving
            // both on keeps an infinite fuel source beside a scarce one — the being
            // would top up at every checkpoint and the stations would never be
            // contested, which is a pressure that is off wearing the costume of one
            // that is on (the `--fuel-auto` denominator bug, ESCADA_DIFICULDADE.md).
            // REGISTERED DECISION (mission rule 7): the flag loses to the degree.
            if SETOR_PAGA.load(Ordering::Relaxed) {
                println!(
                    "  [recurso] per-SECTOR refill disabled — the {n_stations} stations become \
                     the only fuel source (D7 replaces the refill, it does not add to it)."
                );
                SETOR_PAGA.store(false, Ordering::Relaxed);
            }
            // Same reason, for the lap refill: `REFUEL_LAPS` is read at every closed
            // lap. Pushed out of reach instead of adding a branch, so the existing
            // refill code path is untouched.
            REFUEL_LAPS.store(u32::MAX, Ordering::Relaxed);

            // Ceiling, computed here and printed: supply against full demand.
            let tank = FUEL_TANK.load(Ordering::Relaxed) as f64;
            let cap = STATION_CAPACITY.load(Ordering::Relaxed) as f64;
            let regen = STATION_REGEN.load(Ordering::Relaxed) as f64;
            // Cruise consumption at thr ~= 0.7: (0.002 + 0.7^2 * 0.004) / 64 per tick,
            // the same formula the tick charges (context.rs, fuel block).
            let per_tick = (0.002 + 0.7 * 0.7 * 0.004) / 64.0;
            let demand = per_tick * n_ticks as f64 * n_seres as f64;
            // Split, because the total hid the only term the dose controls. Measured
            // 2026-08-26 sweeping stations 16/8/4/2 at charge 0.15: distance moved 2726,
            // 2799, 2981, 2944 m — a 255 m spread against the ~1000 m the flag costs just
            // by being on. The reason is here: `n_seres * tank` is 48.00 u of a 59.40 u
            // demand, 81%, and `--resource-scarce M` cannot touch it. The stations
            // contribute 8.80 u at M=16 down to 1.10 u at M=2, so the whole dial can only
            // move supply between 81% and 96%. It is structurally unable to make the
            // resource scarce; the knob that can is `--fuel-tank`.
            let supply_tanques = n_seres as f64 * tank;
            let supply_postos = n_stations as f64 * (cap + regen * n_ticks as f64);
            let supply = supply_tanques + supply_postos;
            let frac_postos = supply_postos / demand.max(1e-9);
            println!(
                "  [recurso] CEILING before running: supply {supply:.2} u vs full demand \
                 {demand:.2} u = {:.0}% — sustains ~{:.0} of {n_seres} beings for the whole \
                 generation (stations {n_stations}, charge {cap:.2}, tank {tank:.4}).",
                100.0 * supply / demand.max(1e-9),
                supply / (per_tick * n_ticks as f64).max(1e-9),
            );
            println!(
                "  [recurso]   of which starting tanks {supply_tanques:.2} u ({:.0}% of demand, \
                 NOT touched by the dose) and stations {supply_postos:.2} u ({:.0}%) — the \
                 stations are the only part `--resource-scarce` moves.",
                100.0 * supply_tanques / demand.max(1e-9),
                100.0 * frac_postos,
            );
            if supply >= demand {
                println!(
                    "  [recurso] WARNING: supply covers 100% of demand — at this dose the \
                     resource is NOT scarce and the number will match the off arm."
                );
            }
            if frac_postos < 0.25 {
                println!(
                    "  [recurso] WARNING: the stations are only {:.0}% of demand, so changing \
                     their number cannot change much. Starting tanks already cover {:.0}%. \
                     Lower --fuel-tank (or raise --resource-load) before reading M as a dose.",
                    100.0 * frac_postos,
                    100.0 * supply_tanques / demand.max(1e-9),
                );
            }
        }

        if pred_frac > 0.0 {
            let v_pred = pred_frac * TYPICAL_SPEED_MS;
            let d_max = PREDATOR_DRAIN.load(Ordering::Relaxed);
            let cap_life = life_cap() as f32;
            // Fastest possible death: a STOPPED being in contact loses d_max + 1
            // life/tick (the drain plus the ordinary one-per-tick).
            let ticks_to_die = cap_life / (d_max + 1.0).max(1e-6);
            // Physical mortality ceiling: the predator drains one target at a time.
            let max_deaths = (n_ticks as f32 / ticks_to_die.max(1e-6)).floor();
            println!(
                "  [predador] CEILING before running: v_pred {v_pred:.2} m/s vs cruise \
                 {TYPICAL_SPEED_MS:.2} m/s (gap {:+.2} — a straight is an escape while the gap \
                 is >0); fastest possible death {ticks_to_die:.0} ticks from a full cap; \
                 physically possible mortality <= {max_deaths:.0}/{n_seres} = {:.0}%.",
                TYPICAL_SPEED_MS - v_pred,
                100.0 * max_deaths / n_seres as f32,
            );
            if pred_frac >= 1.0 {
                println!(
                    "  [predador] WARNING: dose >= 1.0 — the predator is faster than the \
                     measured cruise, escaping on a straight stops existing, and mortality \
                     becomes luck of the start rather than competence."
                );
            }
        }

        if thermal_gain > 0.0 {
            // Ceiling of the thermal metric: the equilibrium temperature of each
            // sustained regime, closed form, before any run.
            let p = thermal_base_params(thermal_gain);
            let band = universe::tire_thermal::reference_band();
            let dc = universe::tire_thermal::dc_gain(&band, &p);
            println!(
                "  [pneu] CEILING before running: grip floor {:.2}, DC trim {:.3} \
                 (measured DC gain {dc:.4} — 1.000 means the ON arm has the SAME mean grip as \
                 the off arm, so the A/B measures the dynamic and not a nerf).",
                p.floor, p.gain_trim,
            );
            for (nome, slip) in [
                ("parked", 0.00f32),
                ("cruise", 0.05),
                ("proper corner", 0.15),
                ("over-driven", 0.30),
                ("sustained slide", 0.60),
            ] {
                let u = universe::tire_thermal::equilibrium_temp(slip, TYPICAL_SPEED_MS, 1.0, &p);
                println!(
                    "  [pneu]   {nome:<14} slip {slip:.2} → T* {u:.3} ({:.0} C) → grip {:.3}",
                    universe::tire_thermal::celsius(u),
                    universe::tire_thermal::grip_factor(u, &p),
                );
            }
            if (dc - 1.0).abs() > 1e-3 {
                println!(
                    "  [pneu] WARNING: DC gain {dc:.4} != 1.000 — at this dose the ON arm has \
                     a different mean grip from the off arm and the A/B would measure the offset."
                );
            }
        }

        if sensor_dose > 0.0 {
            let k = SENSOR_COST_K.load(Ordering::Relaxed);
            let cap_u = SENSOR_COST_CAP.load(Ordering::Relaxed);
            // The differential of pruning ONE family of 16 channels, in life ticks over
            // a whole generation — the number that says whether selection can see it.
            let dif = sensor_dose * 16.0 / 192.0;
            println!(
                "  [sensor] CEILING before running: tax at reference use = {sensor_dose:.3} \
                 life/tick, maximum {:.3} (u cap {cap_u:.1}), top-K tail = {k}; pruning ONE \
                 family of 16 channels saves {dif:.4} life/tick = {:.0} ticks out of {n_ticks} — \
                 if that is smaller than the typical difference between individuals the pressure \
                 does not select (E-REG1b's negative).",
                sensor_dose * cap_u,
                dif * n_ticks as f32,
            );
        }

        // A pressure the being cannot feel is a pressure that cannot be learned. It is
        // still a legitimate arm (it separates "the world got harder" from "the brain
        // got more input"), so this warns instead of refusing — but it warns LOUDLY,
        // because reading a flat dose curve as "the mechanism is weak" when the signal
        // simply never reached the brain is how a session gets spent.
        if !body_on && (n_stations > 0 || pred_frac > 0.0 || thermal_gain > 0.0) {
            println!(
                "  [aviso] hostile-world pressure ON and --interoception OFF: the being does \
                 NOT feel station/predator/temperature (ch70, ch98, ch114-117, ch124 keep their \
                 old source). Valid arm, but the dose measures the world, not the brain."
            );
        }
        if body_on {
            println!(
                "  [interocepcao] BODY BLOCK on: {} CPU channels -> sensory wave (registry \
                 BODY_SLOTS). ch70/ch71 are a SOURCE MOVE — with the thermal model off they \
                 carry the same number as before.",
                universe::perception::channels::BODY_K
            );
        }
    }
    AUDITAR.store(auditar_visao, Ordering::Relaxed);
    PORTAL_CURVA.store(portal_curva, Ordering::Relaxed);
    SLOW_STEER_AUTH.store(slow_steer_auth, Ordering::Relaxed);
    if life_cap_arg > 0 {
        LIFE_CAP_TICKS.store(life_cap_arg, Ordering::Relaxed);
    } else if life_earn <= 0.0 {
        // Fixed-life episodes initialize `life_left` with `n_ticks`. The body sensor
        // must use the same capacity; the old 1,920 denominator kept ch67 saturated
        // at 1.0 through most long episodes and hid the organism's real remaining life.
        LIFE_CAP_TICKS.store(n_ticks.max(1) as i64, Ordering::Relaxed);
    }
    ctx.brain.out_beta = out_beta;
    ctx.brain.out_gain = ganho_saida;
    ctx.brain.norm_l2_by_std = norm_l2_por_desvio;
    // P1.7 — mesmo bit que congela a mutação em `evolve()` (armado acima, no parsing
    // das flags) TAMBÉM desliga a LEITURA do campo no shader — ver `mech_off_mask`
    // em `BrainGpu`/`brain.wgsl`. Redundante quando o genoma nunca mutou (o campo já
    // está neutro), mas necessário quando `--retomar` carrega um genoma que já tinha
    // mutado ANTES da flag ser passada: sem isto, `--sem-X` deixaria de ser o braço
    // de controle de verdade a partir do segundo `--resume`.
    ctx.brain.mem_off = sem_memoria_neuronal;
    ctx.brain.period_off = sem_periodo_neuronal;
    ctx.brain.threshold_off = sem_limiar_neuronal;
    ctx.brain.activation_off = sem_ativacao_heterogenea;
    if driving_diagnostics_on {
        std::fs::create_dir_all(&out_dir).expect("create diagnostic output directory");
        ctx.driving_diagnostics = Some(
            driving_diagnostics::DrivingDiagnostics::new(
                &ctx.device,
                &ctx.brain,
                std::path::Path::new(&out_dir),
                n_seres,
                vision_being,
                vision_every,
                organism_taxi,
                ablation_plan.as_ref(),
            )
            .unwrap_or_else(|error| {
                eprintln!("ERROR --driving-diagnostics: {error}");
                std::process::exit(1)
            }),
        );
    }
    // `--record-wave`'s config header (Task 5 review round): written here, once, after
    // every flag above that changes the brain's forward pass has its final value, and
    // before the tick loop starts. `set_noise` runs earlier (ruido_sigma/ruido_drop are
    // already in `ctx.brain`, but reading the pre-`set_noise` locals directly is exactly
    // as accurate and avoids relying on the field names staying `pub`).
    //
    // Captures the STARTING configuration, not necessarily every value for the whole
    // run: a multi-generation battery whose difficulty ladder changes noise mid-run
    // (`set_noise` is also called from the ladder, `main.rs:3238`) would need this header
    // re-checked per generation, which `--record-wave`'s existing single-being,
    // single-run design does not need to handle today.
    if let Some(w) = ctx.wave_writer.as_mut() {
        let wave_cfg = universe::gpu::wave_record::WaveConfig {
            dt: 1.0 / 64.0, // gpu_evolve's fixed tick length, see this file's `let dt` sites
            out_beta,
            out_gain: ganho_saida,
            norm_l2_by_std: norm_l2_por_desvio as u32,
            noise_sigma: ruido_sigma,
            dropout_prob: ruido_drop,
            // check-language: off -- existing Portuguese locals from this file's own
            // flag parsing (`sem_memoria_neuronal` etc., already on the allowlist);
            // referencing them here is not new Portuguese prose.
            mem_off: sem_memoria_neuronal as u32,
            period_off: sem_periodo_neuronal as u32,
            threshold_off: sem_limiar_neuronal as u32,
            activation_off: sem_ativacao_heterogenea as u32,
            // check-language: on
            neuromod_regioes: NEUROMOD_REGIOES.load(Ordering::Relaxed) as u32,
            brain_skip,
            pilot_active: (ctx.piloto_forja.is_some() || ctx.piloto_formula.is_some()) as u32,
        };
        if let Err(e) = universe::gpu::wave_record::write_header(w, &wave_cfg) {
            eprintln!("ERROR --record-wave: failed to write the UWAV config header: {e}");
            std::process::exit(1);
        }
    }
    PILOTO_GANHO.store(piloto, Ordering::Relaxed);
    PILOTO_THR.store(
        if piloto_thr > 0.0 { piloto_thr } else { 0.45 },
        Ordering::Relaxed,
    );
    STEER_FIXO.store(steer_fixo, Ordering::Relaxed);
    STEER_ESCALA.store(steer_escala, Ordering::Relaxed);
    PILOTO_SENSOR.store(piloto_sensor, Ordering::Relaxed);
    ATIVACAO_ON.store(ativacao_on, Ordering::Relaxed);
    ATIVACAO_N.store(ativacao_n, Ordering::Relaxed);
    if ativacao_on {
        println!(
            "[ativacao] métricas de ativação L1/L2 LIGADAS — amostra a cada {ativacao_n} ticks"
        );
    }
    if piloto_sensor != 0.0 {
        println!(
            "PILOTO SENSORIAL (ganho {piloto_sensor:+.2}) — sem cérebro, só raios de \
                  parede. Mede o TETO DA INFORMAÇÃO, não o modelo."
        );
    }
    THR_FIXO.store(thr_fixo, Ordering::Relaxed);
    if steer_fixo != 0.0 || thr_fixo != 0.0 {
        println!(
            "OVERRIDE BRUTO: volante={steer_fixo:+.2} acelerador={thr_fixo:.2} \
                  — o cérebro NÃO comanda nada. Isto mede o CORPO."
        );
    }
    if piloto != 0.0 {
        println!(
            "PILOTO FANTASMA ligado (ganho {piloto:+.1}) — o cérebro NÃO comanda o volante. \
                  Isto mede o CORPO, não o modelo."
        );
    }
    if (out_beta - universe::gpu::brain::OUT_BETA_DEFAULT).abs() > 1e-6 {
        println!(
            "fadiga de saída: out_beta={out_beta:.2} (padrão {:.2}) — comando sustentado \
                  sai a {:.0}% da intenção",
            universe::gpu::brain::OUT_BETA_DEFAULT,
            100.0 * (1.0 - out_beta).max(0.0)
        );
    }
    if (ganho_saida - universe::gpu::brain::OUT_GAIN_DEFAULT).abs() > 1e-6 {
        println!(
            "[P1.5] ganho de saída: out_gain={ganho_saida:.2} (padrão {:.2}) — ETAPA 2 \
                  (l2_raw e bypass) escalada antes da fadiga/tanh",
            universe::gpu::brain::OUT_GAIN_DEFAULT
        );
    }
    if norm_l2_por_desvio {
        println!(
            "[P1.5/P1.8] normalização de L2 por desvio-padrão LIGADA (padrão desde P1.8) \
                  — l2_raw dividido pelo desvio dos 48 valores do ser antes do ganho/tanh"
        );
    } else {
        println!(
            "[P1.5/P1.8] normalização de L2 por desvio-padrão DESLIGADA (--sem-norm-l2-por-desvio) \
                  — braço de controle, comportamento histórico pré-P1.8 (l2_std=1.0, saturação de L2 \
                  não corrigida)"
        );
    }
    println!(
        "GpuCtx pronto  [{:.0}ms]",
        t1.elapsed().as_secs_f64() * 1000.0
    );

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
        let latest = latest_checkpoint(&out_dir);
        if let Some((gen_id, path)) = latest {
            println!("Retomando a partir de gen {:4}: {}", gen_id, path);
            if let Some(best) = load_genome(&path) {
                elite_carregado_do_checkpoint = true;
                // Semente a população a partir do melhor genoma + variações.
                let mut pop = Vec::with_capacity(n_seres);
                pop.push(best.clone()); // elite exato
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
                    mix * 100.0,
                    (1.0 - mix) * 100.0
                );
                while pop.len() < n_seres {
                    let mut filho = best.clone();
                    for w in filho.l1.iter_mut() {
                        *w *= mix;
                    }
                    for w in filho.l2.iter_mut() {
                        *w *= mix;
                    }
                    pop.push(filho);
                }
                pop
            } else {
                eprintln!(
                    "[ERROR] --resume: invalid checkpoint {path}; refusing to substitute a random population."
                );
                std::process::exit(1);
            }
        } else {
            // Hard error, not a fallback. Silently swapping the requested champion for a
            // random population is what made a day of measurements meaningless on
            // 2026-08-25: the run reports normally, the numbers look plausible, and
            // nothing in the output says the brain under test is not the one asked for.
            eprintln!(
                "[ERRO] --resume: no evolve_best_gen*.bin found in {out_dir}\n\
                 [ERRO] Refusing to fall back to a random population — that would report\n\
                 [ERRO] plausible numbers for a brain you did not ask for.\n\
                 [ERRO] Copy the checkpoint into the directory, or drop --resume."
            );
            std::process::exit(1);
        }
    } else {
        // População aleatória do zero.
        // Seed = índice do ser (determinístico e diverso).
        (0..n_seres)
            .map(|i| Genome::random(population_seed_for(pop_seed, i)))
            .collect()
    };

    if school_eval_starts > 1 {
        let founders = genomes[..genome_count].to_vec();
        genomes = school_batch::expand_replicas(&founders, school_eval_starts);
    }

    if !evolving_body && genomes.iter().any(|g| g.body != Default::default()) {
        println!(
            "  [body] inherited body genes are preserved but inactive; use --evolving-body to express them"
        );
    }

    // ── FLY MAP: give every being a usable selection ──────────────────────────
    //
    // A genome that came off disk carries no `fly` block at all (a checkpoint written
    // before this feature) or one sized for another map. Either way it packs as INERT —
    // every channel unbound, so the map would command exactly zero for the whole battery
    // and the log would show a dead population with no stated reason. Reseeding at birth
    // values is the honest fix, and it is announced: the rest of that genome is the
    // champion's, only the selection is new.
    if let Some(l) = universe::fly_map::layout() {
        let mut reseeded = 0usize;
        for i in 0..genomes.len() {
            if genomes[i].fly.len() != l.block_len() {
                genomes[i].fly = universe::fly_map::birth_block(population_seed_for(pop_seed, i));
                reseeded += 1;
            }
        }
        if reseeded > 0 {
            println!(
                "  [fly-map] {reseeded}/{} genomes carried no fly block for this map \
                 (checkpoint from before it, or another map) — reseeded at birth values",
                genomes.len()
            );
        }
    }

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
            println!(
                "  [tau-hetero] preservando τ evoluído do elite exato (slot 0) — \
                      banco novo só nos demais {} slots",
                n_seres.saturating_sub(1)
            );
        }
        for (i, g) in genomes.iter_mut().enumerate().skip(start) {
            g.randomize_tau(population_seed_for(pop_seed, i).wrapping_add(0x7A11_0000));
        }
    }

    // Parâmetros evolutivos: configuráveis via CLI, default conservador.
    if tau_semeado && !elite_carregado_do_checkpoint {
        // `tau` follows sensory channels, not the four L1 neuron regions. Seed response
        // bandwidth by signal physics: fast control/proximity, intermediate vision and
        // slow homeostasis. Evolution keeps refining every value from this birth prior.
        for (gi, g) in genomes.iter_mut().enumerate() {
            g.seed_sensory_taus(pop_seed, gi);
        }
        println!(
            "  [tau-seeded] sensory bandwidth prior: control/proximity 0.05-0.40s · vision 0.05-0.80s · homeostasis 0.50-4.00s"
        );
    } else if tau_semeado {
        // A resumed population already inherited a selected time-scale bank. Re-seeding
        // every slot here used to replace that learned organ with eight unrelated banks
        // before generation zero, so an apparent controller comparison was actually a
        // tau randomisation experiment. `--tau-seeded` is a birth prior, not a reset.
        println!("  [tau-seeded] preserving the selected tau bank of the resumed population");
    }
    if mascara_regioes {
        for g in genomes.iter_mut() {
            universe::gpu::evolution::aplicar_mascara_regioes(g);
        }
        println!(
            "  [mascara] regiões DIRIGIDAS ativas (4 blocos de 24: visão/parede/corpo/associativo)"
        );
    }
    // Freeze behaviour-affecting arguments. Only storage/resume controls may differ.
    let mut checkpoint_args = Vec::new();
    let mut arg_index = 1;
    while arg_index < args.len() {
        if ["--out", "--resume-state", "--snapshot-every"].contains(&args[arg_index].as_str()) {
            arg_index += 2;
        } else { checkpoint_args.push(args[arg_index].clone()); arg_index += 1; }
    }
    use sha2::{Digest, Sha256};
    let mut model_hash = Sha256::new();
    for source in [include_bytes!("main.rs").as_slice(), include_bytes!("context.rs").as_slice(),
        include_bytes!("../../gpu/organism.wgsl").as_slice(), include_bytes!("../../gpu/organism.rs").as_slice(),
        include_bytes!("../../gpu/taxi.wgsl").as_slice(), include_bytes!("../../gpu/brain.wgsl").as_slice(),
        include_bytes!("../../gpu/physics_a.wgsl").as_slice(), include_bytes!("../../gpu/physics_b.wgsl").as_slice(),
        include_bytes!("../../gpu/physics_c.wgsl").as_slice(), include_bytes!("../../gpu/organism_reproduction.wgsl").as_slice()] { model_hash.update(source); }
    let checkpoint_config = serde_json::json!({"scenario":"organism-taxi-v1", "arguments":checkpoint_args,
        "beings":n_seres, "tick_rate":64,"implementation_sha256":format!("{:x}",model_hash.finalize())});
    if let Some(path) = resume_state.as_ref() {
        if retomar { panic!("--resume-state and DNA --resume are distinct and mutually exclusive"); }
        let (_, generation, saved_config) = organism_checkpoint::load_metadata(std::path::Path::new(path))
            .expect("checkpoint manifest invalid");
        assert_eq!(saved_config, checkpoint_config, "checkpoint configuration mismatch");
        genomes = organism_checkpoint::load_genomes(std::path::Path::new(path)).expect("checkpoint DNA invalid");
        assert_eq!(genomes.len(), n_seres, "checkpoint population mismatch");
        gen_offset = generation;
    }
    let evo_params = EvoParams {
        p_mutate: evo_p_mutate,
        sigma: evo_sigma,
        motor_p_mutate,
        motor_sigma,
        hip_p: evo_hip_p,
        hip_sigma: evo_hip_sigma,
        fly_p_bind: evo_fly_p_bind,
        fly_p_off: evo_fly_p_off,
        fly_p_on: evo_fly_p_on,
        fly_sigma: evo_fly_sigma,
        n_elite: evo_n_elite,
        tournament_k: evo_tourney,
        congela_l2_cegas,
        mascara_regioes,
        // 5 linhas lidas só quando o câmbio neural dá leitor ao intent[3]/[4];
        // travado/auto: 3 (throttle/steer/brake).
        l2_linhas_lidas: if organism_taxi {
            9
        } else if CAMBIO_MODO.load(Ordering::Relaxed) == 2 {
            5
        } else {
            3
        },
    };

    // ── Histórico de medianas (para inclinação ao final) ──────────────────────
    let mut hist_med: Vec<f64> = Vec::with_capacity(n_gens);
    // Volta da pista em que CADA geração rodou — a tabela final normalizava tudo pela
    // última pista carregada e imprimia V errada em bateria com rotação de currículo
    // (medido: gen28 rodou 9001/2928m, tabela dividiu por 3691m e mostrou 1.256V).
    let mut hist_lap: Vec<f64> = Vec::with_capacity(n_gens);
    let mut hist_max: Vec<f64> = Vec::with_capacity(n_gens);
    let mut hist_results: Vec<GenResult> = Vec::with_capacity(n_gens);

    // Historical runs retain the shared spline. Experimental body/lifetime/ecology
    // runs install per-being inherited heads below, including after reproduction.
    let kan_knots_global = default_kan_knots();

    // ── LAÇO PRINCIPAL ────────────────────────────────────────────────────────
    let t_total = Instant::now();
    if lifetime_flags != 0
        && (ctx.brain_control_direct || ctx.fly.is_some() || ctx.fly_spiking.is_some())
    {
        eprintln!("ERROR lifetime development/plasticity currently requires the native brain");
        std::process::exit(2);
    }
    if ecology_on
        && ((!fuel_on && !organism_taxi)
            || ctx.brain_control_direct
            || ctx.fly_spiking.is_some()
            || ctx.fly.is_some()
            || piloto != 0.0
            || PILOTO_SENSOR.load(Ordering::Relaxed) != 0.0
            || ctx.piloto_formula.is_some()
            || ctx.piloto_forja.is_some()
            || thr_fixo != 0.0
            || steer_fixo != 0.0
            || CUSTO_FIACAO.load(Ordering::Relaxed) > 0.0
            || SENSOR_COST.load(Ordering::Relaxed) > 0.0
            || !ecology_progress.is_finite()
            || ecology_progress < 0.0
            || ecology_founders.is_some_and(|n| n == 0 || n > n_seres))
    {
        eprintln!(
            "ERROR ecology requires fuel, native brain without pilot overrides, disabled wiring/sensor subscription costs, finite nonnegative progress, and 1..population founders"
        );
        std::process::exit(2);
    }
    let mut adaptive = if adaptive_curriculum_on {
        if !mundos.is_empty() || pistas.is_empty() || trocar_ao_completar != 0 {
            eprintln!(
                "ERROR --adaptive-curriculum needs explicit --tracks and cannot combine with --worlds or the legacy curriculum"
            );
            std::process::exit(2);
        }
        Some(
            universe::adaptive_curriculum::AdaptiveCurriculum::new(
                pistas.clone(),
                pista_held_out.into_iter(),
                universe::adaptive_curriculum::CurriculumConfig::default(),
            )
            .unwrap_or_else(|e| {
                eprintln!("ERROR adaptive curriculum: {e:?}");
                std::process::exit(2)
            }),
        )
    } else {
        None
    };
    // School curriculum state (Stage 3): the adaptive arm advances to the next lesson
    // after `school_advance_gens` consecutive generations with p_hat >= school_advance_p;
    // the uniform arm cycles the lesson list. Both see the same generation budget.
    let mut school_lesson_index = 0usize;
    let mut school_streak = 0usize;
    let mut school_last_p_hat: Option<f64> = None;
    for generation_idx in 0..n_gens {
        // gen_abs = número absoluto da geração (para relatório e arquivo de save).
        let gen_abs = gen_offset + generation_idx;
        if let Some(mode) = school_curriculum {
            let mut advanced = false;
            if mode == "adaptive" {
                if let Some(p_hat) = school_last_p_hat.take() {
                    if p_hat >= school_advance_p { school_streak += 1 } else { school_streak = 0 }
                    if school_streak >= school_advance_gens && school_lesson_index + 1 < school_lessons.len() {
                        school_lesson_index += 1;
                        school_streak = 0;
                        advanced = true;
                    }
                }
            } else {
                school_lesson_index = generation_idx % school_lessons.len();
            }
            school_lesson = Some(school_lessons[school_lesson_index]);
            println!(
                "  [curriculum] gen={gen_abs} mode={mode} lesson={} index={school_lesson_index}/{} streak={school_streak} advance={advanced} threshold_p={school_advance_p} threshold_gens={school_advance_gens}",
                school_lessons[school_lesson_index].name(),
                school_lessons.len()
            );
        }
        let genome_diversity = genome_diagnostics::measure(&genomes);
        println!(
            "  [genome] fast hashes {}/{} · RMS(slot0) {:.6} | brake-row hashes {}/{} · RMS(slot0) {:.6}",
            genome_diversity.distinct_fast_hashes,
            n_seres,
            genome_diversity.fast_rms_from_slot0,
            genome_diversity.distinct_brake_row_hashes,
            n_seres,
            genome_diversity.brake_row_rms_from_slot0,
        );
        if let Some(diagnostics) = ctx.driving_diagnostics.as_mut() {
            diagnostics
                .reset(gen_abs, &genomes[diagnostics.selected])
                .unwrap_or_else(|error| {
                    eprintln!("ERROR --driving-diagnostics genome snapshot: {error}");
                    std::process::exit(1)
                });
        }

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
        let rotation_idx = if let Some(curriculum) = adaptive.as_mut() {
            let seed = curriculum
                .seed_for_episode()
                .expect("finite curriculum keeps probing its archive");
            pistas
                .iter()
                .position(|s| *s == seed)
                .expect("training-only curriculum")
        } else if trocar_ao_completar > 0 {
            idx_curriculo
        } else {
            gen_abs
        };
        let (seed_gen, key_gen) = peek_world(&pistas, &mundos, rotation_idx);
        if key_gen != world_key_atual {
            let (world_novo, seed_novo, key_novo) = if organism_taxi {
                let _quiet = universe::world::silence_map_diag();
                resolve_world(&pistas, &mundos, rotation_idx, track_width)
            } else {
                resolve_world(&pistas, &mundos, rotation_idx, track_width)
            };
            world = world_novo;
            if organism_taxi {
                organism_city = Some(universe::organism_city::build_city(&mut world));
                organism_stations = organism_city.as_ref().unwrap().stations.clone();
                universe::organism_scenario::validate_city(&world, &organism_stations).expect("disconnected taxi city");
            }
            ctx.reload_world(&world);
            world_seed_atual = seed_novo;
            world_key_atual = key_novo;
            lap_dist = world.lap_distance as f64;
            let k = FUEL_AUTO_K.load(Ordering::Relaxed);
            if k > 0.0 {
                // REGRA DE TRÊS, AGORA NO DENOMINADOR CERTO (conserto de 2026-08-16).
                //
                // Era X = L/18,75 × 0,003 × k — tanque dimensionado para uma VOLTA
                // inteira. Só que o reabastecimento acontece a cada SETOR
                // (SETOR_PAGA, ver o bloco de checkpoint), e são 24 setores por volta:
                // o tanque saía ~12× maior que o intervalo em que é recarregado, e
                // por isso NUNCA prendia. Medição que expôs: 34 de 35 gerações da
                // bateria-escada com mortalidade 0% mesmo com combustível apertado
                // até k=1,0, ruído, dano e clima LIGADOS ao mesmo tempo.
                //
                // Conta: setor = L/24 ≈ 122 m; a 18,75 m/s são 6,5 s; consumo
                // (0,002 + 0,004·thr²)/s ≈ 0,006/s a fundo → 0,039 por setor. Com
                // tanque de volta (0,468 em 9001) sobrava para 12 setores.
                //
                // Agora k=1,0 significa "exatamente o necessário para alcançar o
                // próximo setor na velocidade típica" — um número que MORDE, e a
                // escada aperta a partir daí. Quem for mais devagar que o típico
                // fica sem combustível: é essa a pressão que faltava.
                let refill_each_sector = SETOR_PAGA.load(Ordering::Relaxed);
                let x = automatic_fuel_tank(lap_dist, k, refill_each_sector);
                FUEL_TANK.store(x, Ordering::Relaxed);
                let interval_name = if refill_each_sector { "setor" } else { "volta" };
                if !organism_taxi {
                    println!(
                        "  [fuel-auto] pista {seed_gen} (volta {lap_dist:.0} m) → tanque {x:.4} \
                              (k={k}; 1,0 = cobre um(a) {interval_name} a 18,75 m/s)"
                    );
                }
            }
        }
        println!(
            "  [{}] gen {:4} usa seed={} ({} waypoints, extensão={:.0}m)",
            if organism_taxi { "city" } else { "pista" },
            gen_abs,
            seed_gen,
            world.waypoints_world.len(),
            lap_dist
        );

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
        let replay_path = std::path::PathBuf::from(&out_dir)
            .join(format!("gen{gen_num:04}.replay", gen_num = gen_abs));
        let _ = std::fs::create_dir_all(&out_dir);
        body_evolution::record(&out_dir, gen_abs, &genomes, evolving_body)
            .expect("write body population provenance");
        // C1 — `hip_gen<N>.f32`: being 0's memory and bias, one row per tick. Opened here
        // so it lands in `--out` with the replay, never next to the source (rule 5 of this
        // project's CLAUDE.md). Small: 13 f32 × ticks ≈ 1 MB at 20 000 ticks.
        ctx.hip_writer = if hippocampus_on() {
            let hip_path = std::path::PathBuf::from(&out_dir)
                .join(format!("hip_gen{gen_num:04}.f32", gen_num = gen_abs));
            match std::fs::File::create(&hip_path) {
                Ok(f) => Some(std::io::BufWriter::new(f)),
                Err(e) => {
                    eprintln!("[hipocampo] could not open {}: {e}", hip_path.display());
                    None
                }
            }
        } else {
            None
        };
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
        let replay_this_gen = generation_idx % replay_every == 0 || generation_idx + 1 == n_gens;
        // Normalize against this mode's actual resource capacity, not the life-earn
        // cap when a fixed-horizon run starts with a different budget.
        let replay_life_capacity = if LIFE_EARN.load(Ordering::Relaxed) > 0.0 {
            life_cap().max(1) as f64
        } else {
            n_ticks.max(1) as f64
        };
        let mut replay_writer = if replay_this_gen {
            let writer = universe::replay::ReplayWriter::create(
                &replay_path,
                universe::replay::SOURCE_CORTEX,
                n_grav as u8,
                64.0,
                seed_gen,
            )
            .unwrap_or_else(|error| {
                eprintln!("ERROR --replay {}: {error}", replay_path.display());
                std::process::exit(1);
            });
            let metadata_path = replay_path.with_extension("meta.json");
            // organism-taxi-v1: context.rs::sample_organism maps body[29] alive to 1/0;
            // the legacy tick budget never exists there, so the replay says so.
            let (replay_life_capacity_meta, life_semantics_meta, life_model_meta) = if organism_taxi {
                (1.0, "alive_flag", "organism-alive-flag: body[29]>0.5 -> 1 else 0")
            } else {
                (replay_life_capacity, "remaining_tick_budget", "legacy-remaining-tick-budget")
            };
            let navigation_observation_meta = organism_taxi.then_some(navigation_observation);
            let mut metadata = serde_json::json!({
                "schema_version": 1,
                "source": "gpu_evolve",
                "tick_rate": 64.0,
                "sample_every": 4,
                "world_seed": seed_gen,
                "generation": gen_abs,
                "wall_cost_per_second": wall_cost,
                "wall_cost_objective_version": 2,
                "retina_road_view_m": universe::visao::road_view_range(),
                "road_view_calibration_version": 2,
                "no_holes": universe::world::NO_HOLES.load(Ordering::Relaxed),
                "terrain": universe::replay::terrain::snapshot(&world),
                "terrain_provenance": "recorded_world_grid",
                "track_width": track_width,
                "frame_phase": "post_physics",
                "sensor_phase": "pre_physics",
                "sensor_source": "captured_cpu_queries",
                "wall_probes_recorded": true,
                "driving_diagnostics": driving_diagnostics_on,
                "fast_memory_erased_each_decision": erase_fast_memory,
                "frame_tick_semantics": "tick4_post_physics_brain64_to_body64",
                "brain_model": if fly_spiking() { "fly_spiking" } else if fly_brain() { "fly_rate" } else { "native" },
                "brain_control": if brain_control_direct { "direct" } else { "legacy" },
                "controller_kind": if taxi_reference_witness { "diagnostic_pilot" } else if piloto != 0.0 || piloto_sensor != 0.0 || ctx.piloto_formula.is_some() || ctx.piloto_forja.is_some() { "diagnostic_pilot" } else { "neural" },
                "planned_ticks": n_ticks,
                "population_seed": pop_seed,
                "checkpoint_loaded": elite_carregado_do_checkpoint,
                "checkpoint_mix": if elite_carregado_do_checkpoint { Some(retomar_mix) } else { None },
                // The replay records being zero. On resume that slot is the exact
                // checkpoint and both tau initialisers deliberately preserve it. The
                // old metadata reported the CLI default (`tau_semeado`) instead of the
                // mutation that actually happened, causing exact champion exports to
                // reject valid checkpoints.
                "tau_reseeded": !elite_carregado_do_checkpoint && (tau_semeado || tau_hetero),
                "start_formation": start_formation,
                "brain_trace_path": fly_spiking_trace_path.as_ref().and_then(|path| std::fs::canonicalize(path).ok()).map(|path| path.to_string_lossy().into_owned()),
                "fuel_enabled": FUEL_ON.load(Ordering::Relaxed),
                "fuel_capacity": FUEL_TANK.load(Ordering::Relaxed),
                "life_capacity": replay_life_capacity_meta,
                "life_semantics": life_semantics_meta,
                "life_frame_field": "hp_norm",
                "arc_semantics": "odometer_m",
                "retina_recorded": false,
                "neural_summary_recorded": false,
                "laps_recorded": true,
                "raw_intents_recorded": replay_this_gen,
            });
            metadata["body_evolution_enabled"] = serde_json::json!(evolving_body);
            metadata["life_model"] = serde_json::json!(life_model_meta);
            metadata["navigation_observation"] = serde_json::json!(navigation_observation_meta);
            metadata["lifetime_flags"] = serde_json::json!(lifetime_flags);
            metadata["evolve_learning"] = serde_json::json!(evolve_learning);
            metadata["adaptive_curriculum"] = serde_json::json!(adaptive_curriculum_on);
            metadata["ecology"] = serde_json::json!(ecology_on);
            metadata["ecology_lineage_scope"] = serde_json::json!(
                "generation; resumed genomes become fresh founders, acquired state is not resumed"
            );
            metadata["body_model"] = serde_json::json!("capacity_mass_tradeoff_v1");
            metadata["body_population_file"] =
                serde_json::json!(format!("body_gen{gen_abs:04}.json"));
            metadata["counterfactual_ablation"] = ablation_plan
                .as_ref()
                .map(|plan| {
                    serde_json::json!({
                        "being": plan.being(),
                        "targets": plan.canonical_spec(),
                        "requested_world_tick": plan.from_tick(),
                        "effective_tick_semantics": "first neural dispatch at or after requested_world_tick"
                    })
                })
                .unwrap_or(serde_json::Value::Null);
            metadata["population_tau_reseeded"] =
                serde_json::json!(if elite_carregado_do_checkpoint {
                    tau_hetero && n_seres > 1
                } else {
                    tau_semeado || tau_hetero
                });
            metadata["motor_head"] = serde_json::json!(if motor_head_linear() {
                "linear"
            } else {
                "spline"
            });
            metadata["slow_kinematics"] = serde_json::json!(if slow_canonical_kinematics() {
                "canonical"
            } else {
                "legacy"
            });
            metadata["wall_probes"] = serde_json::json!(if continuous_wall_probes() {
                "continuous"
            } else {
                "legacy"
            });
            metadata["throttle_arbitration"] =
                serde_json::json!(if throttle_arbitration_signed() {
                    "signed"
                } else {
                    "legacy"
                });
            metadata["reverse_controller"] = serde_json::json!(if reverse_controller_v2() {
                "v2"
            } else {
                "legacy"
            });
            metadata["motor_trace_source"] = serde_json::json!("native_action_mapper_v1");
            metadata["driving_reflex_birth"] = serde_json::json!(driving_reflex_birth);
            metadata["impact_cost_metres_per_kilojoule"] = serde_json::json!(impact_cost);
            metadata["impact_cost_objective_version"] = serde_json::json!(1);
            metadata["damage_scale"] = serde_json::json!(DANO_ESCALA.load(Ordering::Relaxed));
            metadata["impact_energy_to_failure_j"] =
                serde_json::json!(survival::IMPACT_ENERGY_TO_FAILURE_J);
            metadata["life_earn_ticks_per_new_progress_metre"] = if organism_taxi {
                serde_json::Value::Null
            } else {
                serde_json::json!(LIFE_EARN.load(Ordering::Relaxed))
            };
            metadata["fitness_mode"] = serde_json::json!(modo_fitness);
            metadata["refuel_each_sector"] = serde_json::json!(SETOR_PAGA.load(Ordering::Relaxed));
            metadata["refuel_laps"] = serde_json::json!(REFUEL_LAPS.load(Ordering::Relaxed));
            metadata["hole_damage_fraction"] = serde_json::json!(survival::HOLE_DAMAGE_FRACTION);
            let data = serde_json::to_vec_pretty(&metadata).unwrap_or_else(|error| {
                eprintln!(
                    "ERROR --replay metadata {}: {error}",
                    metadata_path.display()
                );
                std::process::exit(1);
            });
            std::fs::write(&metadata_path, data).unwrap_or_else(|error| {
                eprintln!(
                    "ERROR --replay metadata {}: {error}",
                    metadata_path.display()
                );
                std::process::exit(1);
            });
            Some(writer)
        } else {
            None
        };
        ctx.replay_capture = replay_this_gen;
        ctx.replay_raw_motor.clear();
        let t_gen = Instant::now();

        // ── Aplica genomas nos engines ─────────────────────────────────────────
        let mut engines: Vec<SovereignEngine> =
            (0..n_seres).map(|_| SovereignEngine::new()).collect();
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
                ctx.fiacao_custo = genomes
                    .iter()
                    .map(|g| {
                        let (mut num, mut den) = (0.0f64, 0.0f64);
                        for j in 0..universe::config::iara::brain::L1_NEURONS {
                            let pj = j as f64 / 96.0;
                            let linha: Vec<(f64, f64)> = (0
                                ..universe::config::iara::brain::SENSORY_WAVE_SIZE)
                                .map(|c| {
                                    let w = g.l1
                                        [j * universe::config::iara::brain::SENSORY_WAVE_SIZE + c]
                                        .abs() as f64;
                                    let pc = c as f64 / 192.0;
                                    let d = (pj - pc).abs();
                                    (w, d.min(1.0 - d))
                                })
                                .collect();
                            // E-REG1b: com --fiacao-cauda N, só as N conexões mais FORTES do
                            // neurônio pagam fio — é onde os genomas diferem (ver statics.rs).
                            let mut itens = linha;
                            if cauda > 0 {
                                itens.sort_by(|a, b| {
                                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                itens.truncate(cauda);
                            }
                            for (w, d) in itens {
                                num += w * d;
                                den += w;
                            }
                        }
                        (lam as f64 * (num / den.max(1e-9)) * 2.0) as f32
                    })
                    .collect();
            } else {
                ctx.fiacao_custo.clear();
            }
        }
        // ── D8 — sensory subscription per being (`--sensor-cost`) ────────────
        //
        // Same site and same lifetime as the wiring cost above: the weights do not
        // change inside a generation, so neither does the tax; recomputing it per tick
        // would be 18 432 reads × 48 beings × 20 000 ticks for a constant.
        //
        // Why per FAMILY and not per weight — the mechanism, not the intention. E-REG1
        // taxed the mean of 18 432 terms, whose inter-individual variance the CLT
        // drives to zero: the measured spread was 0.2491 vs 0.2493 against a chance
        // value of 0.25, i.e. a constant tax, and a constant tax does not select (same
        // shape as the Landauer control). Here the unit is a family: zeroing the
        // peripheral family removes its whole term, `DOSE * 16/192`, which is a
        // difference a genome can actually own.
        //
        // The tail (top-K per family, K = `--sensor-cost-k`) rather than the family
        // mean, for the same reason E-REG1b moved to a tail: the mean over a family is
        // again an average that concentrates, while the strongest few columns are where
        // genomes differ.
        //
        // `m_i` (mean |w| over the whole layer) is the DC normaliser: without it,
        // shrinking every weight uniformly would cut the tax without pruning anything —
        // the "normalise the DC gain" lesson, in its cheapest possible disguise.
        {
            let dose = SENSOR_COST.load(Ordering::Relaxed);
            if dose > 0.0 {
                use universe::config::iara::brain::{L1_NEURONS, SENSORY_WAVE_SIZE};
                use universe::perception::channels::CHANNEL_FAMILIES;
                let k_tail = SENSOR_COST_K.load(Ordering::Relaxed).max(1) as usize;
                let cap_u = SENSOR_COST_CAP.load(Ordering::Relaxed).max(1e-3);
                // a_F for every being: the top-K mean of the normalised column sums of
                // the family.
                let a_por_ser: Vec<Vec<f32>> = genomes
                    .iter()
                    .map(|g| {
                        let total: f32 = (0..L1_NEURONS * SENSORY_WAVE_SIZE)
                            .map(|w| (g.l1[w] as f32).abs())
                            .sum();
                        let m_i = (total / (L1_NEURONS * SENSORY_WAVE_SIZE) as f32).max(1e-6);
                        let col: Vec<f32> = (0..SENSORY_WAVE_SIZE)
                            .map(|c| {
                                let s: f32 = (0..L1_NEURONS)
                                    .map(|j| (g.l1[j * SENSORY_WAVE_SIZE + c] as f32).abs())
                                    .sum();
                                s / (L1_NEURONS as f32 * m_i)
                            })
                            .collect();
                        CHANNEL_FAMILIES
                            .iter()
                            .map(|&(_, lo, hi)| {
                                let mut v: Vec<f32> = col[lo..hi].to_vec();
                                v.sort_by(|a, b| b.total_cmp(a));
                                let k = k_tail.min(v.len());
                                v[..k].iter().sum::<f32>() / k as f32
                            })
                            .collect()
                    })
                    .collect();
                // Freeze the reference on the first generation the flag is on. Floor
                // eps = 0.05 so a family the population already ignores cannot make
                // `a_F / ref_F` explode (that is what the cap is the second guard for).
                if ref_familias.is_empty() && !a_por_ser.is_empty() {
                    ref_familias = (0..CHANNEL_FAMILIES.len())
                        .map(|f| {
                            let mut col: Vec<f32> = a_por_ser.iter().map(|a| a[f]).collect();
                            col.sort_by(|a, b| a.total_cmp(b));
                            col[col.len() / 2].max(0.05)
                        })
                        .collect();
                    println!(
                        "  [sensor] reference FROZEN at gen {gen_abs} (population median per \
                         family): {}",
                        CHANNEL_FAMILIES
                            .iter()
                            .zip(ref_familias.iter())
                            .map(|(&(nome, _, _), r)| format!("{nome}={r:.2}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
                ctx.sensor_cost = a_por_ser
                    .iter()
                    .map(|a| {
                        let mut c = 0.0f32;
                        for (f, &(_, lo, hi)) in CHANNEL_FAMILIES.iter().enumerate() {
                            let u = (a[f] / ref_familias[f]).min(cap_u);
                            c += ((hi - lo) as f32 / SENSORY_WAVE_SIZE as f32) * u;
                        }
                        dose * c
                    })
                    .collect();
            } else {
                ctx.sensor_cost.clear();
            }
        }
        if NEUROMOD_REGIOES.load(Ordering::Relaxed) {
            ctx.gates_w = genomes
                .iter()
                .flat_map(|g| g.gates_w.iter().copied())
                .collect();
        }
        // C1 — hippocampus: genes flattened once per generation (same lifetime and same
        // reason as `gates_w`), and the 24 slots per being REBUILT, not carried. A new
        // generation is a new life: memory crossing the boundary is the same family of
        // bug as `buf_cfc` inheriting the parent's CfC state, which made the champion's
        // re-evaluation non-deterministic (see this binary's CLAUDE.md, "Recurrent state
        // crosses the generation boundary unless it is reset").
        if hippocampus_on() {
            ctx.hip_genes = genomes.iter().flat_map(|g| g.hip.iter().copied()).collect();
            ctx.hip_mem =
                vec![universe::gpu::hippocampus::SectorMemory::new(); genomes.len().max(n_seres)];
        }
        // FLY MAP: the selection is the only part of this subsystem that changes between
        // generations, and it changes exactly once — same lifetime and same reason as
        // `gates_w` and `hip_genes` above. The state is reset with it: a child inheriting
        // its parent's membrane potentials would make the champion's re-evaluation
        // non-deterministic, which is the bug this binary already paid for with `buf_cfc`.
        if let (Some(fly), Some(l)) = (ctx.fly.as_ref(), universe::fly_map::layout()) {
            let blocks: Vec<&[f32]> = genomes.iter().map(|g| g.fly.as_slice()).collect();
            let mut sel = universe::fly_map::pack_selection(&blocks, l, n_seres);
            // `--fly-readout reflex` (docs/FLY-MAP-BRAIN.md "Reflex readout"): overrides
            // the PACKED `out_pair` with the map's own reflex block, for every being,
            // every generation — `genomes[i].fly` itself (the `Vec<f32>` blocks
            // `pack_selection` just read) is never touched, only this copy.
            if universe::fly_map::readout_reflex_on() {
                if let Some(reflex_out) = fly_map.as_ref().and_then(|m| m.reflex_out) {
                    universe::fly_map::apply_reflex_readout(&mut sel, reflex_out);
                }
            }
            fly.upload_selection(&sel, n_seres);
            fly.reset_state();
            if fly_log_every > 0 && gen_abs % fly_log_every == 0 && !genomes.is_empty() {
                log_fly_elite(gen_abs, &genomes[0].fly, l);
            }
        }
        if let (Some(fly), Some(l)) = (ctx.fly_spiking.as_ref(), universe::fly_map::layout()) {
            let blocks: Vec<&[f32]> = genomes.iter().map(|g| g.fly.as_slice()).collect();
            let mut sel = universe::fly_map::pack_selection(&blocks, l, n_seres);
            if universe::fly_map::readout_reflex_on() {
                if let Some(reflex_out) = fly_map.as_ref().and_then(|m| m.reflex_out) {
                    universe::fly_map::apply_reflex_readout(&mut sel, reflex_out);
                }
            }
            fly.upload_selection(&sel, n_seres);
            fly.reset_state();
            if fly_log_every > 0 && gen_abs % fly_log_every == 0 && !genomes.is_empty() {
                log_fly_elite(gen_abs, &genomes[0].fly, l);
            }
        }
        ctx.upload_engines(&engines);
        // Banco de τ: vai separado porque τ vive no genoma, não no `SovereignEngine`.
        // Uma vez por geração, como os pesos — τ é herdado e fixo na vida do ser.
        ctx.brain.upload_tau(&Genome::pack_taus(&genomes), n_seres);
        ctx.brain.upload_lifetime(
            &genomes.iter().map(|g| g.lifetime).collect::<Vec<_>>(),
            lifetime_flags,
        );
        if organism_taxi || ecology_on || evolving_body || lifetime_flags != 0 {
            ctx.per_being_heads = genomes.iter().map(|g| g.kan_knots).collect();
        }
        // Banco de sensores (FASE SENSORES, degrau 1): mesmo motivo do τ — vive no
        // genoma, sobe uma vez por geração para o pipeline ESPACIAL de percepção.
        // Com `SENSORS_EVOLVE_ON` no padrão (desligado), todo genoma tem
        // `SENSOR_DEFAULTS` e este upload sobe os hardcodes de hoje — equivalente a
        // nunca chamar o upload, só documenta a fonte única.
        ctx.upload_sensors(&Genome::pack_sensors(&genomes), n_seres);
        // P1.7 (cérebro assíncrono) — metadados por neurônio (alfa/período/limiar/
        // ativação): mesmo motivo do τ/sensores, uma vez por geração.
        ctx.brain
            .upload_neuron_meta(&Genome::pack_neuron_meta(&genomes), n_seres);

        // ── P1.7, mecanismo 2 — quantos neurônios de L1 disparam por tick ──────
        //
        // Requisito da tarefa: reportar quantos neurônios EFETIVAMENTE recalculam
        // por tick (média e desvio), com a honestidade de que isto é REPRESENTAÇÃO,
        // não velocidade — ver `simulate_l1_fire_counts`. Simulado em CPU pura (sem
        // GPU, sem readback): o período de cada neurônio já está conhecido antes do
        // tick 0 rodar. Janela de 840 ticks = MMC(1..8), cobre um ciclo completo dos
        // períodos possíveis (1..8, clamp de `evolve()`) independente de onde a
        // amostragem começaria dentro da corrida de verdade.
        if !sem_periodo_neuronal && !organism_taxi {
            let periods_flat: Vec<f32> = genomes
                .iter()
                .flat_map(|g| g.neuron_period_l1.iter().copied())
                .collect();
            let janela = n_ticks.min(840).max(1);
            let (media_disp, desvio_disp) =
                universe::gpu::evolution::simulate_l1_fire_counts(&periods_flat, n_seres, janela);
            let total_neuronios = (n_seres * universe::gpu::evolution::N_L1_NEURONS) as f64;
            println!(
                "  [periodo] neurônios de L1 disparando/tick: {:.1} ± {:.1} de {:.0} ({:.1}%) \
                 · janela simulada {} ticks (REPRESENTAÇÃO, não velocidade — GPU roda todas as threads de qualquer forma)",
                media_disp,
                desvio_disp,
                total_neuronios,
                100.0 * media_disp / total_neuronios.max(1.0),
                janela
            );
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
        // D2: waypoint da GERAÇÃO (mesmo para todos os seres — comparável). Calculado
        // FORA do laço para poder ir ao log: sem ele no log, a geração morta de
        // 2026-08-16 exigiu uma sonda dedicada para descobrir que a largada era o
        // waypoint duplicado do fechamento (ver `frente_do_waypoint`).
        let wi_gen = {
            let h = (gen_abs as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(pop_seed)
                .wrapping_mul(0xFF51AFD7ED558CCD);
            ((h >> 17) as usize) % n_wp
        };
        let formation = if start_formation && !organism_taxi {
            Some(formation_slots(&world, n_seres).unwrap_or_else(|error| {
                panic!("invalid starting formation: {error}");
            }))
        } else {
            None
        };
        let canonical = if !organism_taxi && same_start && formation.is_none() && !largada_aleatoria {
            Some(canonical_start(&world).unwrap_or_else(|error| {
                panic!("invalid canonical start: {error}");
            }))
        } else {
            None
        };
        if let Some(slots) = formation.as_ref() {
            let progress: Vec<String> = slots
                .iter()
                .enumerate()
                .map(|(i, slot)| format!("{i}:{:.2}m", slot.progress_m))
                .collect();
            println!(
                "  [formation] initial progress per being: {}",
                progress.join(" ")
            );
        }
        if largada_aleatoria {
            let wp = world.waypoints_world[wi_gen];
            println!(
                "  [largada] gen {gen_abs:4} nasce no wp {wi_gen}/{n_wp} \
                      (arco ≈ {:.0} m) em ({:.0},{:.0})",
                wi_gen as f64 * lap_dist / n_wp as f64,
                wp.x,
                wp.y
            );
        }
        // School: each being starts on its own junction approach; the lesson geometry
        // is rotated with the same rule as the taxi spawn slots.
        school_shared_pick = None;
        let school_starts = school_lesson.map(|lesson| match school_shared_start {
            Some(mode) => {
                let (starts, index, count) = universe::organism_school::lesson_shared_start_fold(
                    &world,
                    &organism_stations,
                    lesson,
                    &school_contract,
                    n_seres,
                    school_start_fold,
                    mode.pick(gen_abs),
                )
                .expect("city lesson approach for the shared start");
                school_shared_pick = Some((index, count));
                if let Some(g) = starts.first() {
                    println!(
                        "  [school-start] gen {gen_abs:4} shared start {index}/{count} ({}) at ({:.0},{:.0}) target ({:.0},{:.0})",
                        mode.name(),
                        g.start[0],
                        g.start[1],
                        g.target[0],
                        g.target[1]
                    );
                }
                starts
            }
            None => universe::organism_school::lesson_starts_fold(&world, &organism_stations, lesson, &school_contract, n_seres, school_start_fold)
                .expect("city lesson approach capacity"),
        });
        let city_spawns = organism_taxi.then(|| {
            if let Some(starts) = school_starts.as_ref() {
                (0..n_seres)
                    .map(|slot| {
                        let start_index = if school_eval_starts > 1 {
                            school_batch::start_index(
                                slot,
                                school_eval_starts,
                                starts.len(),
                                generation_idx * school_eval_starts,
                            )
                        } else {
                            slot % starts.len()
                        };
                        let g = &starts[start_index];
                        universe::organism_city::CitySpawn {
                        position: [g.start[0], 0.5, g.start[1]],
                        forward: [g.forward[0], 0.0, g.forward[1]],
                        depot_distance_m: g.depot_distance_m,
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                universe::organism_city::spawn_slots(&world, &organism_stations, n_seres).expect("city spawn capacity")
            }
        });
        // Rotate the being -> slot map each generation unless explicitly fixed, so no
        // lineage inherits the slot nearest the waiting passenger.
        let spawn_rotation = if school_eval_starts > 1 {
            // Batch start rotation is already encoded by start_index above. A second
            // slot rotation would break the contiguous genome×start assignment.
            0
        } else if organism_taxi && !fixed_spawn_slots {
            universe::organism_city::slot_rotation(gen_abs as u64, pop_seed, n_seres)
        } else {
            0
        };
        if let Some(spawns) = city_spawns.as_ref() {
            println!(
                "  [spawn] gen {gen_abs:4} rotation {spawn_rotation}: {}",
                (0..n_seres.min(4))
                    .map(|i| {
                        let slot = &spawns[(i + spawn_rotation) % spawns.len()];
                        format!("{i}->({:.0},{:.0}) {:.0}m", slot.position[0], slot.position[2], slot.depot_distance_m)
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        let mut states: Vec<BeingState> = (0..n_seres)
            .map(|i| {
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
                // D2: com --largada-aleatoria, o waypoint da GERAÇÃO vem de hash(gen,seed)
                // — todos os seres da gen largam do mesmo lugar (comparável), mas o lugar
                // muda a cada geração: decorar o traçado a partir da largada deixa de pagar.
                let (pos, fwd) = if let Some(spawns) = &city_spawns {
                    let slot = &spawns[(i + spawn_rotation) % spawns.len()];
                    (slot.position, slot.forward)
                } else if let Some(slots) = formation.as_ref() {
                    let slot = slots[i];
                    (
                        [slot.position.x, 0.0, slot.position.y],
                        [slot.forward.x, 0.0, slot.forward.y],
                    )
                } else if let Some(slot) = canonical {
                    (
                        [slot.position.x, 0.0, slot.position.y],
                        [slot.forward.x, 0.0, slot.forward.y],
                    )
                } else {
                    let wi = if largada_aleatoria {
                        wi_gen
                    } else {
                        (i * n_wp / n_seres).min(n_wp - 1)
                    };
                    let wp = world.waypoints_world[wi];
                    // The final waypoint repeats the first; skip coincident points so
                    // a valid start never receives a zero-length heading.
                    let fwd = frente_do_waypoint(&world.waypoints_world, wi);
                    ([wp.x, 0.0, wp.y], fwd)
                };
                BeingState {
                    phys: body_evolution::at_birth(&genomes[i], evolving_body, pos, fwd),
                    prev_steer: 0.0,
                    stagnation_ticks: 0,
                    wall_ticks: 0,
                    collision_energy_j: 0.0,
                    impact_count: 0,
                    hole_hits: 0,
                    hole_cooldown_ticks: 0,
                    center_prog: 0.0,
                    sensor_steer: 0.0,
                    sensor_thr: 0.0,
                    sensor_brk: 0.0,
                    wp_near: 0,
                    fuel: body_evolution::tank_capacity(
                        &genomes[i],
                        evolving_body,
                        FUEL_TANK.load(Ordering::Relaxed),
                    ),
                    fuel_capacity: body_evolution::tank_capacity(
                        &genomes[i],
                        evolving_body,
                        FUEL_TANK.load(Ordering::Relaxed),
                    ),
                    ticks_alive: 0,
                    // Earned-life runs start with a finite 30-second exploration budget.
                    // New ordered progress must buy every additional second.
                    life_left: if LIFE_EARN.load(Ordering::Relaxed) > 0.0 {
                        LIFE_START as f64
                    } else {
                        n_ticks as f64
                    },
                    reverse_timer: 0.0,
                    post_reverse_lockout: 0.0,
                    shift_cooldown: 0.0,
                    shift_up_prev: 0.0,
                    shift_down_prev: 0.0,
                    nm_reserva: [1.0; 4],
                    re_falhas: 0,
                    re_desde_saida: 999.0,
                    dano: 0.0,
                    // Ambient, not zero: `step_temp` clamps to `[ambient, 1.0]`, so a
                    // tyre born at 0 would jump to ambient on its first tick and the
                    // first tick of every generation would read as a transient that is
                    // not there. Unused while `--tire-thermal` is off.
                    tire_temp: universe::tire_thermal::DEFAULT_AMBIENT,
                    joy_tick: 0.0,
                    learning_reward: 0.0,
                    joy_streak: 0.0,
                    edge_row6: 0.0,
                    edge_row7: 0.0,
                    phantom_d_min: 30.0,
                    dist_wall_l: 40.0,
                    dist_wall_r: 40.0,
                    dist_wall_f: 50.0,
                }
            })
            .collect();

        // ── HOSTILE WORLD: per-generation setup (BODY PHASE, 2026-08-17) ───────
        //
        // Everything here is rebuilt every generation, for the same reason the
        // recurrent brain state is reset a few lines above: state that survives the
        // generation boundary makes the champion's re-evaluation non-deterministic. A
        // half-drained station field or a predator still standing where it caught
        // someone last generation is exactly that, and it is invisible in every
        // aggregate the report prints.
        //
        // Determinism: the station seed and the predator pose are functions of
        // `(gen_abs, pop_seed, seed_gen)` only, so every being of a generation meets
        // the SAME world, and two runs with the same seeds meet the same world too —
        // the comparability contract the shared start (D2) already relies on.
        {
            ctx.reset_hostile_telemetry();

            // D7 — stations.
            let m_postos = STATIONS_N.load(Ordering::Relaxed) as usize;
            ctx.station_cfg = universe::resources::StationConfig {
                capacity: STATION_CAPACITY.load(Ordering::Relaxed),
                recharge_per_tick: STATION_REGEN.load(Ordering::Relaxed),
                radius_m: STATION_RADIUS_M.load(Ordering::Relaxed),
                transfer_per_tick: STATION_TRANSFER.load(Ordering::Relaxed),
            };
            ctx.stations = if m_postos > 0 {
                let seed_postos = (gen_abs as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(pop_seed)
                    ^ seed_gen;
                universe::resources::place_stations(
                    seed_postos,
                    m_postos,
                    &world.waypoints_world,
                    &world.waypoint_arcs,
                    track_width,
                    ctx.station_cfg.capacity,
                )
            } else {
                Vec::new()
            };

            // PREDATOR — spawned on the track axis, `--predator-spawn` metres of ARC
            // behind where the generation starts.
            //
            // Behind, and on the axis, and not on a seed-derived circle around the
            // start: with `--random-start` the start moves every generation, and a
            // predator placed by a raw circle would sometimes land in front of the
            // pack. "In front" is not a harder dose, it is a different mechanism (a
            // head-on obstacle), and mixing the two makes the dose curve unreadable.
            let pred_frac = PREDATOR_SPEED_FRAC.load(Ordering::Relaxed);
            ctx.predator_params = universe::predator::PredatorParams {
                speed_cap: pred_frac * TYPICAL_SPEED_MS,
                catch_radius_m: PREDATOR_RADIUS_M.load(Ordering::Relaxed),
                ..Default::default()
            };
            ctx.predator_grace_ticks = PREDATOR_GRACE_TICKS.load(Ordering::Relaxed) as usize;
            ctx.predator = if pred_frac > 0.0 {
                let arcs = &world.waypoint_arcs;
                let lap = arcs.last().copied().unwrap_or(1.0).max(1e-3);
                let wi_start = if largada_aleatoria { wi_gen } else { 0 };
                let arc_start = arcs.get(wi_start).copied().unwrap_or(0.0);
                let back = PREDATOR_SPAWN_BACK_M.load(Ordering::Relaxed);
                let s = (arc_start - back).rem_euclid(lap);
                let (pos, tangent) =
                    universe::resources::point_at_arc(&world.waypoints_world, arcs, s);
                Some(universe::predator::Predator {
                    pos,
                    vel: bevy::math::Vec2::ZERO,
                    // Facing along the track, i.e. toward the start line it is behind.
                    heading: tangent,
                })
            } else {
                None
            };

            // TYRE THERMAL — one parameter set per being, each DC-normalised on its own.
            //
            // Heterogenising a parameter WITHOUT re-normalising its DC gain is a mistake
            // already paid for here: a being whose optimum sits away from the band the
            // track visits is simply slower on average, so selection picks the PARAMETER
            // instead of the behaviour and the knob becomes a handicap wearing a
            // diversity costume. `heterogeneous_normalized` does both in one call.
            let thermal_gain = TIRE_THERMAL_GAIN.load(Ordering::Relaxed);
            ctx.thermal = if thermal_gain > 0.0 {
                let base = thermal_base_params(thermal_gain);
                let band = universe::tire_thermal::reference_band();
                (0..n_seres)
                    .map(|i| {
                        universe::tire_thermal::ThermalParams::heterogeneous_normalized(
                            pop_seed,
                            i,
                            universe::tire_thermal::DEFAULT_OPTIMAL_SPREAD,
                            &base,
                            &band,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };
        }

        let birth_positions: Vec<_> = states
            .iter()
            .map(|s| (s.phys.pos, s.phys.fwd, s.wp_near))
            .collect();
        let mut ecology = if ecology_on {
            let base_tank = if organism_taxi { 5_000_000.0 } else { FUEL_TANK.load(Ordering::Relaxed) as f64 };
            for state in states.iter_mut().skip(ecology_founders.unwrap_or(n_seres)) {
                state.life_left = 0.0;
                state.fuel = 0.0;
            }
            let runtime = ecology_runtime::EcologyRuntime::new(
                universe::ecology::EcologyConfig {
                    capacity: n_seres,
                    maturity_ticks: ecology_maturity,
                    reproduction_cooldown_ticks: ecology_maturity.max(64),
                    reproduction_cost: 0.05 * base_tank,
                    child_energy: 0.2 * base_tank,
                    parent_energy_reserve: 0.1 * base_tank,
                    minimum_progress: ecology_progress,
                    seed: pop_seed ^ gen_abs as u64,
                    max_events: n_seres * 4,
                },
                n_seres,
            )
            .expect("valid ecology configuration");
            let observer_factory = if resume_state.is_some() { ecology_runtime::GenotypeBirthObserver::resume } else { ecology_runtime::GenotypeBirthObserver::create };
            let observer = observer_factory(
                format!("{out_dir}/lineage_gen{gen_abs:04}.jsonl"),
                format!("{out_dir}/births_gen{gen_abs:04}"),
            )
            .expect("create lineage observer");
            Some(runtime.with_observer(Box::new(observer)))
        } else {
            None
        };

        // ── Odômetro ───────────────────────────────────────────────────────────
        let mut dist_acum: Vec<f64> = vec![0.0; n_seres];
        // Progresso AO LONGO DA PISTA, monotônico e com tratamento de cruzamento de linha.
        //
        // A fitness era o hodômetro puro (`dist_acum`), que conta qualquer deslocamento:
        // fora da pista, em círculo, indo e voltando. Seguir o circuito nunca deu vantagem
        // nenhuma — e o replay mostrou o campeão indo RETO e saindo da pista na primeira
        // curva. Ele não errava a curva; fazer a curva não valia nada.
        let mut prog_acum: Vec<f64> = vec![0.0; n_seres];
        // Survival uses a separate high-water mark. It pays a newly conquered track
        // metre once, never odometer motion or repeated forward/backward oscillation.
        let mut life_progress_max: Vec<f64> = vec![0.0; n_seres];
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
        // D6: cruzamentos (setor, tick) por ser — vira o fantasma da próxima gen.
        let mut cruzamentos: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_seres];
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
                if ang > 8.0 {
                    v.push(acc);
                }
            }
            v
        };
        if generation_idx == 0 && !organism_taxi {
            println!(
                "  [portais] {} portais em curvas > 8° (uniforme seriam {})",
                portais.len(),
                N_CHECKPOINTS
            );
        }
        let mut voltas_reais: Vec<u32> = vec![0; n_seres];
        // C0 (`laps.rs`): per-lap ledger of THIS generation. `voltas_reais` counts laps;
        // this records what each one cost — ticks per sector, metres, grass, damage.
        let mut lap_tracker = laps::LapTracker::new(n_seres);
        let mut prog_prev: Vec<f32> = (0..n_seres)
            .map(|i| world.get_track_progress(states[i].phys.pos[0], states[i].phys.pos[2]))
            .collect();
        let mut pos_prev: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

        // Warm-up tick (JIT compilation). Its side effects (CfC state, the
        // P1.7 period counters, `--observe`'s and
        // `--record-wave`'s row 0) are not, and reach every recording this run produces.
        // Before Task 5 this line was merely misleading; it is now load-bearing: the
        // open-loop screen's bit-exact parity (`screen.rs`) depends on row 0 being
        // recorded here and replayed there, because the P1.7 per-neuron recompute
        // counter only advances once per brain dispatch. `--observe`/`--record-wave`
        // record this tick deliberately — do not "fix" it to discard its side effects
        // without re-deriving screen.rs's warm-up replay, or parity breaks silently.
        let mut school_aggregates: Option<Vec<school_batch::SchoolAggregate>> = None;
        if organism_taxi {
            // `school_bench` is the default mode *when a school lesson exists*.
            // Passing it unconditionally makes the ordinary taxi world look like
            // a bench and disables passenger issuance in `taxi.wgsl`.
            let taxi_mode = if school_bench && school_lesson.is_some_and(|lesson| lesson.is_service()) {
                TaxiMode::ServiceBench
            } else if school_lesson.is_some() && school_bench {
                TaxiMode::MotorBench
            } else {
                TaxiMode::Production
            };
            ctx.start_organism(&genomes, &organism_stations, &organism_city.as_ref().unwrap().traffic_lights(), if ecology_on { ecology_founders.unwrap_or(n_seres) } else { n_seres }, taxi_mode)
                .expect("organism initialization failed");
            if taxi_reference_witness {
                ctx.taxi_witness = Some(universe::gpu::taxi_witness::TaxiWitnessGpu::new(&ctx.device, ctx.capacity));
            }
            ctx.taxi.as_mut().unwrap().configure_routes(&ctx.device, &world, &organism_stations).expect("organism road routes");
            if let Some(starts) = school_starts.as_ref() {
                // Bench: passengers off (the judge owns the task slots). Life: passengers on.
                ctx.taxi.as_mut().unwrap().set_mode(if school_bench
                    && school_lesson.is_some_and(|lesson| lesson.is_service())
                {
                    TaxiMode::ServiceBench
                } else if school_bench {
                    TaxiMode::MotorBench
                } else {
                    TaxiMode::Production
                });
                if ctx.school.is_none() {
                    ctx.school = Some(universe::gpu::school::SchoolGpu::new(
                        &ctx.device,
                        ctx.capacity,
                        school_contract,
                        school_horizon_ticks,
                        school_bench,
                        school_stop_shaping,
                    ));
                }
                let rotated: Vec<_> = (0..n_seres)
                    .map(|slot| {
                        let index = if school_eval_starts > 1 {
                            school_batch::start_index(
                                slot,
                                school_eval_starts,
                                starts.len(),
                                generation_idx * school_eval_starts,
                            )
                        } else {
                            (slot + spawn_rotation) % starts.len()
                        };
                        starts[index]
                    })
                    .collect();
                ctx.school.as_ref().unwrap().reset(&ctx.queue, &rotated).expect("school lesson reset");
            }
            ctx.sample_organism(&mut states).expect("invalid organism genesis");
            let spawn_slots_json = city_spawns.as_ref().map(|slots| {
                slots
                    .iter()
                    .enumerate()
                    .map(|(k, slot)| {
                        serde_json::json!({"slot": k, "x_m": slot.position[0], "z_m": slot.position[2],
                            "forward": [slot.forward[0], slot.forward[2]], "depot_distance_m": slot.depot_distance_m})
                    })
                    .collect::<Vec<_>>()
            });
            let spawn_assignment_json = serde_json::json!({
                "mode": if fixed_spawn_slots { "fixed" } else { "rotated_by_hash_generation_seed" },
                "rotation": spawn_rotation,
                "being_to_slot": "(being + rotation) % n",
                "contract": "lane centre, flow heading, >=18 m from stations, >=8 m apart, 4.5 m from walls",
            });
            let manifest = serde_json::json!({
                "scenario": universe::organism_scenario::SCENARIO_ID,
                "observation": universe::organism_scenario::observation_id(navigation_observation),
                "navigation_observation": navigation_observation,
                "spawn_slots": spawn_slots_json,
                "spawn_assignment": spawn_assignment_json,
                "action": universe::organism_scenario::ACTION_ID,
                "task_potential": universe::gpu::taxi_routes::ROAD_POTENTIAL_VERSION,
                "controller": if taxi_reference_witness {universe::gpu::taxi_witness::TAXI_WITNESS_ID} else {"evolved-native"},
                "learning_evidence": !taxi_reference_witness,
                "generation": gen_abs, "tick_rate": 64, "stations": organism_stations,
                "geometry": universe::organism_scenario::export_world_geometry(&world),
                "city": organism_city.as_ref().unwrap().metadata(),
                "sensory_channels": universe::organism_channels::channel_manifest(navigation_observation),
                "traffic_observation": {"schema":universe::gpu::traffic::TRAFFIC_OBSERVATION_ID,"wave_channels":{"present":172,"distance_fraction_100m":173,"phase":182,"time_remaining_fraction":191}},
                "body_state_owner": "GPU", "body_sample_stride": 64,
                "sensor_extras": "organism-resident-extras-v1",
                "action_channels": ["throttle", "steer", "brake", "gear_up", "gear_down", "service", "pickup", "dropoff", "reverse"],
                "action_stride": 96, "active_action_channels": 9,
                "pedal_arbitration": "exclusive_signed_net_v1",
                "static_rest": "bounded_contact_v1",
                "simulation_transfer": "resident GPU; host observation on sampling boundaries",
                "fitness": "1000*deliveries + bounded_task_progress + money + 0.001*age_seconds + 10*alive - 100*(1-integrity) - 100*(1-alive)",
                "motor_controller": if taxi_reference_witness {"reference GPU controller"} else {"evolved policy"}, "resource_units": "joules,kilograms",
                "sensory_remap": if navigation_observation == 3 {"160..175 interoception;176..191 taxi v3: body-frame direction cos/sin + d/(d+100) for task and service"} else {"160..175 interoception;176..191 taxi;explicit GPS destination"}});
            let mut manifest = manifest;
            manifest["feedback_mode"] = serde_json::json!(match feedback_mode { 1 => "symmetric", 2 => "progress", _ => "legacy" });
            manifest["online_reward"] = serde_json::json!(match feedback_mode {
                1 => "clamp(delivered - 2*damage + 0.01*energy_change + 0.01*min(0, money_change/25), -1, 1)",
                2 => "clamp(delivered - 2*damage + 0.01*energy_change + 0.01*min(0, money_change/25) + 0.01*max(0, same-key task progress gain), -1, 1)",
                _ => "clamp(delivered - 2*damage + 0.2*max(0,energy_change) + 0.01*min(0,energy_change), -1, 1)",
            });
            manifest["plasticity_flags"] = serde_json::json!(lifetime_flags);
            manifest["plasticity_freeze_after"] = serde_json::json!(plasticity_freeze_after);
            manifest["task_progress"] = serde_json::json!("phase-aware: approach 0..100 on best road distance closed toward the nearest waiting passenger (key=id*2), carrying 100 + 0..100 toward the destination (key=id*2+1); memory in body 31/61/62");
            if let (Some(lesson), Some(starts)) = (school_lesson, school_starts.as_ref()) {
                let rotated: Vec<_> = (0..n_seres)
                    .map(|slot| {
                        let index = if school_eval_starts > 1 {
                            school_batch::start_index(
                                slot,
                                school_eval_starts,
                                starts.len(),
                                generation_idx * school_eval_starts,
                            )
                        } else {
                            (slot + spawn_rotation) % starts.len()
                        };
                        starts[index].to_json()
                    })
                    .collect();
                manifest["school"] = serde_json::json!({
                    "id": universe::gpu::school::SCHOOL_ID, "task_id": lesson.name(), "lesson_id": lesson.id(), "version": universe::gpu::school::SCHOOL_SCHEMA_VERSION,
                    "mode": if school_bench { "bench" } else { "life" }, "env_seed": pop_seed, "start_tick": 0,
                    "horizon_ticks": school_horizon_ticks, "tick_rate": 64,
                    "observable_objective": "sub-goal direction cos/sin (178/179), d/(d+100) (183), has_task (180); v2: dx/dz per 100 m (178/179)",
                    "initial_condition": "lane centre before a junction, heading along the flow, one approach per being",
                    "assistance": "none", "body_version": universe::organism_scenario::SCENARIO_ID,
                    "isolation": if school_bench { "taxi-taxi contacts off; passengers off; retina unchanged" } else { "none" },
                    "tolerances": school_contract.to_json(),
                    "success": {"move": "dist(pos,target)<=goal_radius and propelled_ticks>=propelled_min", "stop": "speed>=v_min before the area, then speed<=v_stop inside for hold_ticks while alive", "turn": "entered the junction box, then dist(pos,exit)<=gate_radius with dot(heading,exit_dir)>=align_min"},
                    "failure_codes": {"1": "contact", "2": "death", "3": "corridor", "4": "born_inside_target", "5": "overshoot", "6": "drove_through_junction", "7": "wrong_exit", "8": "no_approach_speed"},
                    "truncation": "status 3 at horizon, progress kept",
                    "score": if school_stop_shaping { "1000*success + 500*progress - 200*failed - 100*(1-integrity) - 100*(1-alive) + stop: 300*clamp(1 - min_speed_inside/v_min) + 200*clamp(max_near_stop_dwell/hold_ticks), near-stop=2*v_stop; success remains v_stop" } else { "1000*success + 500*progress - 200*failed - 100*(1-integrity) - 100*(1-alive)" },
                    "stop_shaping": school_stop_shaping,
                    "fitness": if school_bench { "score" } else { "taxi fitness (life mode)" }, "lessons": rotated,
                    "curriculum": school_curriculum.unwrap_or("fixed"),
                    "curriculum_lessons": school_lessons.iter().map(|l| l.name()).collect::<Vec<_>>(),
                    "curriculum_index": school_lesson_index, "advance_p": school_advance_p, "advance_gens": school_advance_gens,
                    "start_fold": school_start_fold.name(),
                    "start_assignment": if school_eval_starts > 1 { "same_deterministic_batch_per_genome" } else { school_shared_start.map_or("per_slot_rotated_by_hash", |m| m.name()) },
                    "shared_start": school_shared_pick.map(|(index, count)| serde_json::json!({"index": index, "count": count})),
                    "eval_starts_per_genome": school_eval_starts,
                    "genome_count": genome_count,
                    "physical_slots": n_seres,
                    "start_candidates": universe::organism_school::lesson_start_candidates(&world, &organism_stations, lesson, &school_contract).len(),
                });
                if school_bench {
                    manifest["fitness"] = serde_json::Value::String("school score (see school)".to_string());
                }
            }
            std::fs::write(
                format!("{out_dir}/organism_gen{gen_abs:04}.json"),
                serde_json::to_vec_pretty(&manifest).unwrap(),
            )
            .expect("organism manifest write");
        }
        if !organism_taxi && !ctx.tick(&mut states, n_seres, 1.0 / 64.0, 0, &kan_knots_global) {
            eprintln!("[ERROR] warmup failed; no valid generation result can be reported");
            std::process::exit(1);
        }

        let mut first_tick = 0;
        if generation_idx == 0 {
            if let Some(path) = resume_state.as_ref() {
                let (tick, generation, aux) = organism_checkpoint::load(&mut ctx, &mut states,
                    std::path::Path::new(path), &checkpoint_config).expect("checkpoint restore failed");
                assert_eq!(generation, gen_abs);
                assert!(tick <= n_ticks, "checkpoint beyond requested horizon");
                first_tick = tick;
                macro_rules! restore_vector { ($name:ident) => {
                    $name = serde_json::from_value(aux[stringify!($name)].clone()).expect("checkpoint episode vector");
                    assert_eq!($name.len(), n_seres, "checkpoint episode length");
                }; }
                restore_vector!(dist_acum); restore_vector!(prog_acum); restore_vector!(prog_prev);
                restore_vector!(pos_prev); restore_vector!(checkpoint); restore_vector!(voltas_reais);
                restore_vector!(life_progress_max);
                restore_vector!(cruzamentos);
                lap_tracker = serde_json::from_value(aux["lap_tracker"].clone()).expect("checkpoint lap tracker");
                if let Some(runtime) = ecology.as_mut() {
                    runtime.restore_checkpoint(serde_json::from_value(aux["ecology"].clone()).expect("checkpoint ecology")).expect("invalid ecology state");
                }
                ctx.sample_organism(&mut states).expect("restored organism invalid");
                if let Some(diagnostics) = ctx.driving_diagnostics.as_mut() {
                    let ids: Vec<u64> = ecology.as_ref().map(|e| e.lineage_ids().to_vec()).unwrap_or_else(|| (1..=n_seres as u64).collect());
                    let positions: Vec<_> = states.iter().map(|s| s.phys.pos).collect();
                    diagnostics.resume_identity(tick, &ids, &genomes, &positions).expect("resume trace identity");
                }
            }
        }
        // ── N ticks de simulação ───────────────────────────────────────────────
        let dt = 1.0_f32 / 64.0;
        let mut executed_ticks = first_tick;
        let mut episode_end_reason = "horizon";
        for tick_i in first_tick..n_ticks {
            ctx.force_organism_sample = tick_i + 1 == n_ticks || (snapshot_every > 0 && (tick_i + 1) % snapshot_every == 0);
            if plasticity_freeze_after == Some(tick_i) {
                // From this dispatch on the residual P is applied but never updated;
                // `upload_lifetime` at the next generation restores the learning flags.
                ctx.brain.set_lifetime_flags(lifetime_flags | 8);
                println!("  [plasticity] gen {gen_abs} frozen at tick {tick_i}: flags {}", ctx.brain.lifetime_flags());
            }
            let ok = ctx.tick(&mut states, n_seres, dt, tick_i + 1, &kan_knots_global);
            if !ok {
                eprintln!(
                    "[gen {:4}] ERROR at tick {}; invalidating the run.",
                    gen_abs, tick_i
                );
                std::process::exit(1);
            }
            // Odômetro: distância euclidiana entre pos[t-1] e pos[t].
            // Com dt=1/64s e v≤60m/s, passo ≤0.94m → erro ≈ arclength < 0.1%.
            // Quadro de replay do líder, a 16 Hz.
            if (tick_i + 1) % 4 == 0 {
                if let Some(ref mut rw) = replay_writer {
                    for lider in 0..n_grav {
                        let ph = &states[lider].phys;
                        let vel = (ph.vel[0].powi(2) + ph.vel[2].powi(2)).sqrt();
                        let raw = ctx
                            .replay_raw_motor
                            .get(lider)
                            .copied()
                            .unwrap_or([f32::NAN; 3]);
                        if raw.iter().any(|v| !v.is_finite()) {
                            eprintln!(
                                "ERROR --replay: missing/nonfinite raw motor command for being {lider} at tick {}",
                                tick_i + 1
                            );
                            std::process::exit(1);
                        }
                        let fuel_norm = if FUEL_ON.load(Ordering::Relaxed) {
                            let tank = (states[lider].fuel_capacity as f32).max(0.01);
                            ((states[lider].fuel as f32 / tank).clamp(0.0, 1.0) * 127.0) as u8
                        } else {
                            0
                        };
                        let rf = universe::replay::ReplayFrame {
                            pos: ph.pos,
                            fwd: [ph.fwd[0], ph.fwd[2]],
                            speed: vel,
                            yaw_rate: ph.av_y,
                            arc_prog: dist_acum[lider] as f32,
                            laps: voltas_reais[lider].min(u16::MAX as u32) as u16,
                            gear: ph.current_gear as u8,
                            // Vida restante no lugar do HP: é o recurso que este caminho
                            // realmente consome, e é o que explica o ser parar.
                            hp_norm: ((states[lider].life_left.max(0.0) as f32
                                / replay_life_capacity as f32)
                                .min(1.0)
                                * 127.0) as u8,
                            fuel_norm,
                            tick: (tick_i + 1) as u32,
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
                            intent_thr: raw[0],
                            intent_str: raw[1],
                            intent_brk: raw[2],
                            ..Default::default()
                        };
                        if let Err(error) = rw.write_frame(lider as u8, &rf) {
                            eprintln!("ERROR --replay {}: {error}", replay_path.display());
                            std::process::exit(1);
                        }
                    }
                }
            }

            for i in 0..n_seres {
                // Organism state and odometry are advanced exclusively by resident
                // kernels. Legacy lap rewards/host body feedback do not apply to a city.
                if organism_taxi {
                    if let Some(body) = ctx.organism_samples.get(i) {
                        dist_acum[i] = body.values[98] as f64;
                    }
                    pos_prev[i] = states[i].phys.pos;
                    continue;
                }
                let p1 = pos_of(&states[i].phys);
                let dx = (p1[0] - pos_prev[i][0]) as f64;
                let dz = (p1[2] - pos_prev[i][2]) as f64;
                if states[i].life_left > 0.0 {
                    let passo = (dx * dx + dz * dz).sqrt();
                    dist_acum[i] += passo;
                    // C0: one tick charged to the sector this being is in. Inside the
                    // `life_left > 0` guard on purpose — a corpse is pinned to
                    // `throttle=0, brake=1` and its ticks are not lap time. It follows
                    // the odometer step so `dist_acum` is the SAME reading `on_lap` below
                    // gets at a crossing; that pair (tick, odometer) is where the partial
                    // lap `LapTracker::finish` writes ends, and a corpse never files one
                    // past its own last tick.
                    lap_tracker.on_tick(
                        i,
                        tick_i as u32,
                        dist_acum[i] as f32,
                        ctx.lap_grass[i],
                        ctx.lap_damage[i],
                        ctx.lap_dry[i],
                    );

                    // Avanço na pista desde o tick anterior, com o salto da linha de
                    // chegada tratado nos dois sentidos. O clamp a 10% da volta descarta
                    // teleporte numérico sem descartar avanço legítimo.
                    let cur = world.get_track_progress(p1[0], p1[2]);
                    let ld = lap_dist as f32;
                    let mut d = cur - prog_prev[i];
                    if d < -ld * 0.5 {
                        d += ld;
                    } else if d > ld * 0.5 {
                        d -= ld;
                    }
                    let avanco = d.clamp(-ld * 0.1, ld * 0.1) as f64;
                    prog_acum[i] += avanco;
                    states[i].learning_reward = (avanco * 64.0 / 30.0).clamp(-1.0, 1.0) as f32;
                    prog_prev[i] = cur;

                    // DOPAMINA — canais 126 (joy_tick) e 127 (joy_streak).
                    //
                    // Replicando motor.rs:1666-1677: joy_tick = progress_speed *
                    // surface_factor, joy_streak acumula em 5 s de flow e decai 2× rápido.
                    // progress_speed = speed alinhada / 30 m/s; surface_factor depende do
                    // voxel (estrada=1, grama=0.2, parede=-0.5). O ser deve SENTIR quando
                    // está indo bem para que o cérebro lento (System 2) possa modular.
                    {
                        let p = &states[i].phys;
                        let spd = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                        let vel_dot = p.vel[0] * p.fwd[0] + p.vel[2] * p.fwd[2];
                        let align = if spd > 0.1 {
                            (vel_dot / spd).clamp(-1.0, 1.0)
                        } else {
                            1.0
                        };
                        let progress_speed = (spd * align.max(0.0) / 30.0).clamp(0.0, 1.0);
                        // Voxel sob o ser usando a grade CPU (mesma grade usada para
                        // propriocepção acima — não há custo adicional).
                        let gs = ctx.grid_size_cpu as usize;
                        let cs = ctx.cell_size_cpu;
                        let xi = (p.pos[0] / cs).round() as i32;
                        let zi = (p.pos[2] / cs).round() as i32;
                        let vox = if xi >= 0 && xi < gs as i32 && zi >= 0 && zi < gs as i32 {
                            ctx.grid_cpu[(zi as usize) * gs + xi as usize]
                        } else {
                            2u32
                        };
                        // 0=grama, 1=estrada, 2=parede, 3=buraco (VoxelType ordem em world.rs)
                        let surface_factor: f32 = match vox {
                            1 => 1.0,  // Road
                            0 => 0.2,  // Grass
                            2 => -0.5, // Wall
                            _ => -1.0, // Hole (3) e demais
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
                        let life_capacity = if LIFE_EARN.load(Ordering::Relaxed) > 0.0 {
                            life_cap().max(1) as f64
                        } else {
                            n_ticks.max(1) as f64
                        };
                        let state = &mut states[i];
                        if !organism_taxi
                            && survival::apply_hole_contact(
                                &mut state.life_left,
                                &mut state.hole_cooldown_ticks,
                                vox == 3,
                                life_capacity,
                            )
                        {
                            state.hole_hits = state.hole_hits.saturating_add(1);
                            state.joy_streak = 0.0;
                        }
                        // C1 — hippocampus: one tick of evidence about the sector being
                        // traversed. Written HERE and not at the crossing because this is
                        // where `vox` and `spd` already exist; a second copy of the
                        // voxel lookup would be a second thing to keep in sync
                        // (`docs/STYLE.md` §4), and there is no extra cost.
                        //
                        // All three inputs are REAL, none is a placeholder: `spd` is the
                        // speed this tick, `vox == 0` is grass (VoxelType order in
                        // `world.rs`, the same table `surface_factor` reads above), and
                        // `states[i].dano` is D4's hidden damage, already normalised to
                        // [0, 1] — `SectorMemory::accumulate` stores the DIFFERENCE
                        // across the sector, so the slot says "this corner hurt" instead
                        // of "I am an old car". With `--damage 0` (the default) the component is
                        // a constant 0, which is the truth for that run, not an invention.
                        if let Some(m) = ctx.hip_mem.get_mut(i) {
                            m.accumulate(spd, vox == 0, states[i].dano);
                        }
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
                        let dd = dxx * dxx + dzz * dzz;
                        if dd < best_d {
                            best_d = dd;
                            melhor = idx;
                        }
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
                        portais
                            .iter()
                            .filter(|&&p| cur >= p)
                            .count()
                            .saturating_sub(1) as u32
                    } else {
                        sector_of(cur, ld)
                    };
                    let total = if PORTAL_CURVA.load(Ordering::Relaxed) {
                        n_port
                    } else {
                        N_CHECKPOINTS
                    };
                    // C1 — hippocampus READ index: the GEOMETRIC sector, updated every
                    // tick. Not `checkpoint[i]`: the ratchet only advances on the next
                    // sector in sequence, so a being that went backwards or skipped one
                    // would be handed the memory of a sector it is not in. The write side
                    // does follow the ratchet — see `SectorMemory::read_sector`.
                    if let Some(m) = ctx.hip_mem.get_mut(i) {
                        m.set_read_sector(setor);
                    }
                    let esperado = checkpoint[i] % total.max(1);
                    if setor == esperado {
                        checkpoint[i] += 1;
                        // D6: registra o cruzamento p/ virar fantasma da próxima gen; e
                        // se há fantasma ativo, paga o bônus de estar à frente (moeda
                        // cruzada: VIDA; assimétrico — atrás não perde, ver fantasma.rs).
                        cruzamentos[i].push((checkpoint[i] % N_CHECKPOINTS.max(1), tick_i as u32));
                        // C0: same event. `setor` is the sector the being actually
                        // entered, not the `checkpoint + 1` index the ghost stores.
                        lap_tracker.on_sector(i, setor, tick_i as u32);
                        // C1 — hippocampus WRITE: close the sector just traversed, open
                        // the one just entered. `cross` writes the sector the being is
                        // LEAVING, never `setor`, because the accumulated evidence
                        // describes the stretch already driven and `read` expects
                        // `mem[s]` to mean "what happened while I was in s".
                        //
                        // DISCREPANCY WITH THE PRD, recorded not resolved:
                        // docs/PRD-CEREBRO-NEUROMOD.md §C1 describes this site as
                        // "`cruzamentos[i]` receives `(setor, tick)`". It does not — the
                        // line above pushes `checkpoint[i] % 24` AFTER the increment,
                        // which is `(setor + 1) % 24`, the NEXT expected sector. That is
                        // the ghost's own convention (`fantasma.rs:94-99`) and is left
                        // alone; the hippocampus takes `setor` directly and does not read
                        // `cruzamentos`, so the two are independent.
                        //
                        // α comes from the genome (`sigmoid(genes[27])`, 0.5 at birth),
                        // per being — with the organ off the bank is empty and this is a
                        // `None` match that costs one bounds check.
                        if hippocampus_on() {
                            let alpha = ctx
                                .hip_genes
                                .get(
                                    i * universe::gpu::hippocampus::N_HIP_GENES
                                        ..(i + 1) * universe::gpu::hippocampus::N_HIP_GENES,
                                )
                                .map(universe::gpu::hippocampus::write_rate)
                                .unwrap_or(0.5);
                            if let Some(m) = ctx.hip_mem.get_mut(i) {
                                m.cross(setor, alpha);
                            }
                        }
                        if !organism_taxi && fantasma_on {
                            let v = fantasma_atual
                                .vantagem(checkpoint[i] % N_CHECKPOINTS.max(1), tick_i as u32);
                            states[i].life_left = (states[i].life_left
                                + fantasma_atual.bonus_vida(v) as f64)
                                .min(life_cap() as f64);
                        }
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
                        if !organism_taxi
                            && FUEL_ON.load(Ordering::Relaxed)
                            && SETOR_PAGA.load(Ordering::Relaxed)
                        {
                            states[i].fuel = states[i].fuel_capacity;
                        }
                        if checkpoint[i] % total.max(1) == 0 {
                            voltas_reais[i] += 1;
                            // C0: a lap closes where `voltas_reais` says it closes —
                            // one definition of a lap in this binary, not two.
                            lap_tracker.on_lap(i, tick_i as u32, dist_acum[i] as f32);
                            // FECHAR VOLTA REABASTECE — o incentivo que faltava.
                            // Na f1-world era a cada 3 voltas; aqui a cada volta, porque
                            // partindo de ZERO voltas exigir três seria um degrau que a
                            // evolução nunca sobe. O intervalo é `--refuel-laps`.
                            if !organism_taxi
                                && voltas_reais[i] % REFUEL_LAPS.load(Ordering::Relaxed) == 0
                            {
                                states[i].fuel = states[i].fuel_capacity;
                            }
                        }
                    }
                    let ganho = LIFE_EARN.load(Ordering::Relaxed);
                    if !organism_taxi && ganho > 0.0 {
                        let earned = survival::life_gain_from_new_progress(
                            prog_acum[i],
                            &mut life_progress_max[i],
                            ganho as f64,
                        );
                        // The cap prevents one fast straight from buying unlimited future.
                        states[i].life_left = (states[i].life_left + earned).min(life_cap() as f64);
                    }
                }
                pos_prev[i] = p1;
            }

            if let Some(ecology) = ecology.as_mut() {
                let birth_life = if LIFE_EARN.load(Ordering::Relaxed) > 0.0 {
                    LIFE_START as f64
                } else {
                    n_ticks.max(1) as f64
                };
                let mutation_flags = (if evolve_learning { 1 } else { 0 }) | (lifetime_flags & 6);
                let births = if organism_taxi {
                    if (tick_i + 1) % 64 == 0 {
                        let poses: Vec<_> = states.iter().map(|s| (s.phys.pos, s.phys.fwd, s.wp_near)).collect();
                        let task_progress: Vec<_> = ctx.organism_samples.iter().map(|s| s.values[68] as f64).collect();
                        ecology.step_organism(&mut ctx, &mut states, &mut genomes, &mut engines,
                            &task_progress, &evo_params, evolving_body, mutation_flags, (tick_i + 1) as u64,
                            |_slot, parent, genome| {
                                let (pos, fwd, wp_near) = poses[parent];
                                ecology_runtime::SpawnState {
                                    physics: body_evolution::at_birth(genome, evolving_body, pos, fwd),
                                    fuel_capacity: 5_000_000.0,
                                    life_ticks: 1.0, wp_near,
                                }
                            }).unwrap_or_else(|e| panic!("organism ecological birth: {e:?}"))
                    } else { Vec::new() }
                } else { ecology
                    .step(
                        &mut ctx,
                        &mut states,
                        &mut genomes,
                        &mut engines,
                        &prog_acum,
                        &evo_params,
                        evolving_body,
                        mutation_flags,
                        |slot, genome| {
                            let (pos, fwd, wp_near) = birth_positions[slot];
                            ecology_runtime::SpawnState {
                                physics: body_evolution::at_birth(genome, evolving_body, pos, fwd),
                                fuel_capacity: body_evolution::tank_capacity(
                                    genome,
                                    evolving_body,
                                    FUEL_TANK.load(Ordering::Relaxed),
                                ),
                                life_ticks: birth_life,
                                wp_near,
                            }
                        },
                    )
                    .unwrap_or_else(|e| {
                        eprintln!("ERROR ecological birth: {e:?}");
                        std::process::exit(1)
                    }) };
                for birth in births {
                    let slot = birth.slot;
                    dist_acum[slot] = 0.0;
                    prog_acum[slot] = 0.0;
                    life_progress_max[slot] = 0.0;
                    checkpoint[slot] = 0;
                    cruzamentos[slot].clear();
                    voltas_reais[slot] = 0;
                    prog_prev[slot] = world
                        .get_track_progress(states[slot].phys.pos[0], states[slot].phys.pos[2]);
                    pos_prev[slot] = states[slot].phys.pos;
                    lap_tracker.reset_slot(slot, tick_i as u32 + 1);
                    if let Some(diagnostics) = ctx.driving_diagnostics.as_mut() {
                        diagnostics
                            .on_birth(
                                slot,
                                birth.child_lineage_id,
                                &genomes[slot],
                                states[slot].phys.pos,
                            )
                            .expect("record newborn observation provenance");
                    }
                    println!(
                        "  [birth] tick={} slot={} parent={} lineage={} fuel={:.6}",
                        tick_i + 1,
                        slot,
                        birth.parent_slot,
                        birth.child_lineage_id,
                        states[slot].fuel
                    );
                }
            }

            // ── Painel: snapshot de tick (E5, a cada 64 ticks) ────────────────────
            // Fica DEPOIS do laço por-ser de cima de propósito: states[i].life_left/fuel
            // já refletem este tick (vida ganha, combustível reabastecido no setor, etc.).
            if let Some(p) = &painel {
                if tick_i % 64 == 0 {
                    let snaps: Vec<painel::BeingSnapshot> = (0..n_seres)
                        .map(|i| {
                            let ph = &states[i].phys;
                            let vel = (ph.vel[0].powi(2) + ph.vel[2].powi(2)).sqrt();
                            painel::BeingSnapshot {
                                id: i,
                                x: ph.pos[0],
                                z: ph.pos[2],
                                vel,
                                life: (states[i].life_left.max(0.0) as f32 / life_cap() as f32)
                                    .min(1.0),
                                vivo: states[i].life_left > 0.0,
                            }
                        })
                        .collect();
                    let fuel_mediana = {
                        let mut fv: Vec<f64> = states
                            .iter()
                            .take(n_seres)
                            .map(|s| s.fuel.max(0.0))
                            .collect();
                        fv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        fv[fv.len() / 2] as f32
                    };
                    p.update_tick(tick_i as u64, snaps, fuel_mediana);
                }
            }
            if organism_taxi && snapshot_every > 0 && (tick_i + 1) % snapshot_every == 0 {
                let path = std::path::PathBuf::from(format!("{out_dir}/state-gen{gen_abs:04}-tick{:08}",tick_i+1));
                let aux = serde_json::json!({"dist_acum":dist_acum,"prog_acum":prog_acum,"prog_prev":prog_prev,
                    "pos_prev":pos_prev,"checkpoint":checkpoint,"voltas_reais":voltas_reais,"life_progress_max":life_progress_max,
                    "cruzamentos":cruzamentos,"lap_tracker":lap_tracker,"ecology":ecology.as_ref().map(|e| e.checkpoint())});
                organism_checkpoint::save(&ctx, &states, &genomes, &path, tick_i+1, gen_abs,
                    checkpoint_config.clone(), aux).expect("runtime checkpoint save failed");
            }
            executed_ticks = tick_i + 1;
            if organism_taxi && executed_ticks % 64 == 0 && ctx.organism_samples.iter().all(|s| s.values[29] == 0.0) {
                episode_end_reason = "extinction";
                break;
            }
        }
        let n_ticks = executed_ticks;
        if organism_taxi {
            std::fs::write(format!("{out_dir}/organism_episode_gen{gen_abs:04}.json"),
                serde_json::json!({"schema_version":1,"end_reason":episode_end_reason,"ticks_executed":executed_ticks,
                    "tick_rate":64,"scope":"mechanical execution; learning requires multi-seed evaluation"}).to_string()).expect("episode outcome write");
        }

        if let Some(diagnostics) = ctx.driving_diagnostics.as_mut() {
            diagnostics
                .flush_window(n_ticks, &states)
                .expect("flush driving evidence");
        }
        if let Some(mut writer) = replay_writer.take() {
            if let Err(error) = writer.flush() {
                eprintln!("ERROR --replay {}: {error}", replay_path.display());
                std::process::exit(1);
            }
        }

        // ── Custo de parede ────────────────────────────────────────────────────
        // Subtrai da fitness o tempo colado na parede, convertido em metros. Com
        // `--wall-cost 0` nada muda e a linha vira só telemetria: é a condição de controle
        // do experimento pareado, no MESMO binário, sem recompilar entre condições.
        let wall_frac: f64 = {
            let total: u64 = states
                .iter()
                .take(n_seres)
                .map(|s| s.wall_ticks as u64)
                .sum();
            total as f64 / (n_seres as f64 * n_ticks as f64)
        };

        // ── Estatísticas da geração ────────────────────────────────────────────
        // A fitness que a evolução enxerga. Progresso por padrão; `--fitness dist`
        // reproduz o hodômetro antigo, para comparar em vez de assumir.
        let mut fitness: Vec<f64> = match modo_fitness.as_str() {
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
            "centro" => (0..n_seres)
                .map(|i| {
                    let sector_count = if PORTAL_CURVA.load(Ordering::Relaxed) {
                        portais.len()
                    } else {
                        N_CHECKPOINTS as usize
                    };
                    let ordered = objective_cost::ordered_progress_metres(
                        checkpoint[i],
                        prog_acum[i],
                        lap_dist,
                        sector_count,
                    );
                    objective_cost::centered_progress_metres(ordered, states[i].center_prog)
                })
                .collect(),
            "checkpoint" => (0..n_seres)
                .map(|i| {
                    let sector_count = if PORTAL_CURVA.load(Ordering::Relaxed) {
                        portais.len()
                    } else {
                        N_CHECKPOINTS as usize
                    };
                    objective_cost::ordered_progress_metres(
                        checkpoint[i],
                        prog_acum[i],
                        lap_dist,
                        sector_count,
                    )
                })
                .collect(),
            "checkpoint-legacy" => (0..n_seres)
                .map(|i| checkpoint[i] as f64 * lap_dist + prog_acum[i].max(0.0))
                .collect(),
            "centro-legacy" => (0..n_seres)
                .map(|i| checkpoint[i] as f64 * lap_dist + states[i].center_prog.max(0.0))
                .collect(),
            _ => unreachable!("fitness mode validated before GPU setup"),
        };

        if organism_taxi {
            ctx.sample_organism(&mut states)
                .expect("organism final state invalid");
            let rows: Vec<_> = ctx
                .organism_samples
                .iter()
                .enumerate()
                .map(|(slot, body)| {
                    let v = &body.values;
                    serde_json::json!({"slot":slot,"tick":executed_ticks,"traffic":ctx.traffic_samples.get(slot),"age_seconds":v[3],"alive":v[29],
                    "energy_j":v[0],"water_kg":v[1],"material_kg":v[2],"engine_temp_c":v[9]-273.15,
                    "engine_wear":v[6],"tire_wear":v[7],"integrity":v[8],"maturity":v[4],
                    "fatigue":v[5],"distance_m":v[98],"collision_energy_j":v[100],"impact_ticks":v[101],
                    "brake_ticks":v[102],"task_progress_score":v[127],
                    "money":v[64],"deliveries":v[68],"paid_repair_j":v[11],"paid_growth_j":v[12],
                    "task_key":v[31],"task_best_remaining_m":v[61],"task_initial_remaining_m":v[62],
                    "fitness_components":{"events":1000.0*v[68],"task_progress":v[127],"money":v[64],
                        "age":0.001*v[3],"alive_bonus":10.0*v[29],"integrity_penalty":-100.0*(1.0-v[8]),
                        "death_penalty":-100.0*(1.0-v[29]),"total":universe::organism_scenario::fitness(body)}})
                })
                .collect();
            std::fs::write(
                format!("{out_dir}/organism_result_gen{gen_abs:04}.json"),
                serde_json::to_vec_pretty(&rows).unwrap(),
            )
            .expect("organism results write");
            fitness = ctx
                .organism_samples
                .iter()
                .map(universe::organism_scenario::fitness)
                .collect();
            if let (Some(lesson), Some(school)) = (school_lesson, ctx.school.as_ref()) {
                use universe::gpu::school::*;
                let records = school.download(&ctx.device, &ctx.queue, n_seres).expect("school records");
                let rows: Vec<_> = records
                    .iter()
                    .enumerate()
                    .map(|(slot, r)| {
                        let v = &ctx.organism_samples[slot].values;
                        let mut row = record_json(r, v[0], v[98]);
                        row["slot"] = serde_json::json!(slot);
                        row["genome"] = serde_json::json!(slot / school_eval_starts);
                        row["replica_start"] = serde_json::json!(slot % school_eval_starts);
                        row["alive"] = serde_json::json!(v[29] > 0.5);
                        row["integrity"] = serde_json::json!(v[8]);
                        row["impacts_lifetime"] = serde_json::json!(v[101]);
                        row["collision_energy_lifetime_j"] = serde_json::json!(v[100]);
                        row["refusal_code"] = serde_json::json!(v[69]);
                        row["contact_ticks_lifetime"] = serde_json::json!(v[99]);
                        row["elite_same_genome"] = serde_json::json!(slot == 0 && generation_idx > 0);
                        if let Some(starts) = school_starts.as_ref() {
                            let start_index = if school_eval_starts > 1 {
                                school_batch::start_index(
                                    slot,
                                    school_eval_starts,
                                    starts.len(),
                                    generation_idx * school_eval_starts,
                                )
                            } else {
                                (slot + spawn_rotation) % starts.len()
                            };
                            let g = &starts[start_index];
                            row["start_index"] = serde_json::json!(start_index);
                            row["start"] = serde_json::json!([g.start[0], g.start[1]]);
                            row["forward"] = serde_json::json!([g.forward[0], g.forward[1]]);
                        }
                        row
                    })
                    .collect();
                let record_vectors: Vec<Vec<f32>> = records.iter().map(|record| record.to_vec()).collect();
                let aggregates = school_batch::aggregate_school_batch(
                    &record_vectors,
                    genome_count,
                    school_eval_starts,
                );
                let aggregate_rows: Vec<_> = aggregates
                    .iter()
                    .map(|aggregate| serde_json::json!({
                        "genome": aggregate.genome,
                        "selection_score": aggregate.score,
                        "mean_score": aggregate.mean_score,
                        "worst_score": aggregate.worst_score,
                        "worst_failure_penalty": aggregate.worst_failure_penalty,
                        "successes": aggregate.successes,
                        "failures": aggregate.failures,
                        "deaths": aggregate.deaths,
                        "starts": aggregate.starts,
                    }))
                    .collect();
                let count = |status: u32| records.iter().filter(|r| r[R_STATUS] as u32 == status).count();
                let attempts = n_seres;
                let successes = count(STATUS_SUCCESS);
                let mut by_code = serde_json::Map::new();
                for code in 1..=15u32 {
                    let k = records.iter().filter(|r| r[R_STATUS] as u32 == STATUS_FAILED && r[R_FAIL_CODE] as u32 == code).count();
                    if k > 0 { by_code.insert(fail_code_name(code).to_string(), serde_json::json!(k)); }
                }
                let mut progress: Vec<f32> = records.iter().map(|r| r[R_PROGRESS]).collect();
                progress.sort_by(|a, b| a.total_cmp(b));
                let mut scores: Vec<f32> = records.iter().map(|r| r[R_SCORE]).collect();
                scores.sort_by(|a, b| a.total_cmp(b));
                let deaths = ctx.organism_samples.iter().filter(|b| b.values[29] <= 0.5).count();
                let summary = serde_json::json!({
                    "generation": gen_abs, "lesson": lesson.name(), "horizon_ticks": school_horizon_ticks,
                    "attempts": attempts, "successes": successes, "failures": count(STATUS_FAILED),
                    "truncated": count(STATUS_TRUNCATED), "running": count(STATUS_RUNNING),
                    "failures_by_code": by_code, "deaths": deaths,
                    "competence_p_hat": (successes as f64 + 1.0) / (attempts as f64 + 2.0),
                    "progress_median": progress[progress.len() / 2], "progress_mean": progress.iter().sum::<f32>() / progress.len() as f32,
                    "score_median": scores[scores.len() / 2], "score_max": scores[scores.len() - 1],
                    "energy_spent_mean_j": rows.iter().map(|r| r["energy_spent_j"].as_f64().unwrap_or(0.0)).sum::<f64>() / attempts as f64,
                    "contact_ticks_total": records.iter().map(|r| r[R_CONTACT_TICKS]).sum::<f32>(),
                    "elite_same_genome": generation_idx > 0,
                    "elite_status": status_name(records[0][R_STATUS] as u32), "elite_progress": records[0][R_PROGRESS],
                    "mode": if school_bench { "bench" } else { "life" },
                    "curriculum": school_curriculum.unwrap_or("fixed"), "curriculum_index": school_lesson_index,
                    "start_fold": school_start_fold.name(),
                    "start_assignment": school_shared_start.map_or("per_slot_rotated_by_hash", |m| m.name()),
                    "shared_start": school_shared_pick.map(|(index, count)| serde_json::json!({"index": index, "count": count})),
                    "genome_count": genome_count, "eval_starts_per_genome": school_eval_starts,
                    "aggregates": aggregate_rows,
                    "beings": rows,
                });
                school_last_p_hat = Some((successes as f64 + 1.0) / (attempts as f64 + 2.0));
                std::fs::write(
                    format!("{out_dir}/school_result_gen{gen_abs:04}.json"),
                    serde_json::to_vec_pretty(&summary).unwrap(),
                )
                .expect("school results write");
                if school_bench {
                    fitness = records.iter().map(|r| r[R_SCORE] as f64).collect();
                }
                school_aggregates = Some(aggregates);
                let pickup_requests = records.iter().map(|r| r[R_PICKUP_REQUESTS] as u64).sum::<u64>();
                let dropoff_requests = records.iter().map(|r| r[R_DROPOFF_REQUESTS] as u64).sum::<u64>();
                let refusals = records.iter().filter(|r| r[R_REFUSAL_CODE] > 0.5).count();
                println!(
                    "  [school] generation={gen_abs} lesson={} success={successes}/{attempts} failed={} truncated={} deaths={deaths} pickup_requests={pickup_requests} dropoff_requests={dropoff_requests} refusals={refusals} p_hat={:.3} progress_median={:.3} score_median={:.1} score_max={:.1} codes={}",
                    lesson.name(), count(STATUS_FAILED), count(STATUS_TRUNCATED),
                    (successes as f64 + 1.0) / (attempts as f64 + 2.0),
                    progress[progress.len() / 2], scores[scores.len() / 2], scores[scores.len() - 1],
                    serde_json::Value::Object(by_code.clone())
                );
            }
            let sample_count = ctx.organism_samples.len().max(1) as f64;
            println!("{}", driving_diagnostics::taxi_generation_summary(
                gen_abs,
                ctx.organism_samples.iter().filter(|b| b.values[29] > 0.5).count(),
                n_seres,
                ctx.organism_samples.iter().filter(|b| b.values[90] > 0.5).count(),
                ctx.organism_samples.iter().filter(|b| b.values[65] > 0.5).count(),
                ctx.organism_samples.iter().map(|b| b.values[68] as u64).sum::<u64>(),
                ctx.organism_samples.iter().filter(|b| b.values[69] > 0.5).count(),
                ctx.organism_samples.iter().map(|b| b.values[101].max(0.0) as u64).sum(),
                ctx.organism_samples.iter().map(|b| b.values[127] as f64).sum::<f64>() / sample_count,
                ctx.organism_samples.iter().map(|b| b.values[0] as f64).sum::<f64>() / sample_count,
                ctx.organism_samples.iter().map(|b| b.values[1] as f64).sum::<f64>() / sample_count,
            ));
        }

        // Safety discounts useful progress instead of subtracting into a zero floor.
        // A subtract-and-clamp objective made both failure modes optimal in sequence:
        // first crashing was free, then standing still beat every negative score.
        // The relative form keeps motion necessary while letting clean motion win.
        // The curriculum gate, when armed, holds the costs at zero until the population's
        // MEDIAN uncosted fitness reaches the threshold. While the gate is shut `fitness`
        // is untouched, so the median read below is the uncosted one by construction.
        // The reading is taken BEFORE the gate is allowed to open and the costs are applied
        // BEFORE it is re-read, in that order, so the generation that crosses the threshold
        // is still scored without the cost and the pressure starts on the next one. Writing
        // it the other way round -- opening the gate first -- charged the crossing
        // generation too, which the six-generation smoke caught by comparing champion
        // genomes against the no-cost arm.
        let costs_apply = cost_gate_progress.is_none() || cost_gate_open;
        let uncosted_median = cost_gate_progress
            .filter(|_| !cost_gate_open)
            .map(|_| objective_cost::population_median(&fitness[..n_seres]));
        if !organism_taxi && costs_apply && (wall_cost > 0.0 || impact_cost > 0.0) {
            let sector_metres = lap_dist / N_CHECKPOINTS as f64;
            for (score, state) in fitness.iter_mut().zip(&states) {
                *score = objective_cost::safety_adjusted_score(
                    *score,
                    state.wall_ticks,
                    wall_cost,
                    state.collision_energy_j,
                    impact_cost,
                    sector_metres,
                );
            }
        }
        if let (Some(threshold), Some(median)) = (cost_gate_progress, uncosted_median) {
            if median >= threshold {
                cost_gate_open = true;
                println!(
                    "[cost-gate] gen {generation_idx} median {median:.1} m >= {threshold:.1} m: \
                     wall/impact cost applies from the next generation"
                );
            }
        }
        let selection_fitness: Vec<f64> = school_aggregates
            .as_ref()
            .map(|aggregates| aggregates.iter().map(|aggregate| aggregate.score).collect())
            .unwrap_or_else(|| fitness.clone());

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
                let agora = selection_fitness[0];
                let razao = if anterior > 1.0 {
                    agora / anterior
                } else {
                    f64::NAN
                };
                println!(
                    "  [elite] mesmo genoma: {:.0} → {:.0} (fitness, {:.0}% do anterior) · hodômetro={:.0}m",
                    anterior,
                    agora,
                    100.0 * razao,
                    dist_acum[0]
                );
                elite_reprod.push(razao);
            } else {
                println!(
                    "  [elite] pista mudou desde a última reavaliação (seed {} → {}) — razão \
                     não comparável, pulando",
                    elite_fitness_anterior_pista.unwrap_or(0),
                    seed_gen
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
            let mut f = selection_fitness.clone();
            f.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (lo, hi) = (f[0], f[f.len() - 1]);
            let m = f.iter().sum::<f64>() / f.len() as f64;
            let sd = (f.iter().map(|v| (v - m).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
            let distintos = {
                let mut u = f.clone();
                u.dedup_by(|a, b| (*a - *b).abs() < 0.5);
                u.len()
            };
            println!(
                "  [seleção] fitness {:.0}..{:.0} · desvio {:.2} · valores distintos {}/{}",
                lo,
                hi,
                sd,
                distintos,
                f.len()
            );
            if distintos <= 2 {
                println!("            ⇒ SELEÇÃO CEGA: quase todos com a mesma nota.");
            }
        }

        // gen_stats conta "voltas" como fitness/lap_dist. Com a fitness de checkpoint
        // valendo lap_dist por setor, isso vira número inflado (3 setores = "3 voltas").
        // As VOLTAS mostradas têm de vir do progresso real na pista, não da fitness.
        let mut r = gen_stats(
            gen_abs,
            &if organism_taxi {dist_acum.clone()} else {prog_acum.iter().map(|p| p.max(0.0)).collect::<Vec<_>>()},
            lap_dist,
        );
        let t_gen_s = t_gen.elapsed().as_secs_f64();

        // Record every being, separating physical death from low track progress.
        let population: Vec<_> = (0..n_seres)
            .map(|i| {
                serde_json::json!({
                    "being": i, "progress_m": if organism_taxi {None} else {Some(prog_acum[i])}, "distance_m": dist_acum[i],
                    "scenario": if organism_taxi {"organism-taxi-v1"} else {"race-v1"},
                    "deliveries": if organism_taxi {Some(ctx.organism_samples[i].values[68])} else {None},
                    "fitness": fitness[i], "alive_ticks": states[i].ticks_alive,
                    "impact_count": if organism_taxi {ctx.organism_samples[i].values[101] as u32} else {states[i].impact_count},
                    "collision_energy_j": if organism_taxi {ctx.organism_samples[i].values[100] as f64} else {states[i].collision_energy_j},
                    "hole_hits": if organism_taxi {None} else {Some(states[i].hole_hits)},
                    "fuel_remaining": states[i].fuel,
                    "life_remaining_ticks": if organism_taxi {None} else {Some(states[i].life_left)},
                    "low_progress": if organism_taxi {dist_acum[i] < 10.0} else {prog_acum[i].max(0.0) < 10.0},
                    "life_exhausted": states[i].life_left <= 0.0,
                    "x": states[i].phys.pos[0], "z": states[i].phys.pos[2],
                })
            })
            .collect();
        let population_path = format!("{out_dir}/population_gen{gen_abs:04}.json");
        std::fs::write(
            &population_path,
            serde_json::to_vec_pretty(&population).unwrap(),
        )
        .expect("write population measurement artifact");
        println!(
            "  [population] low_progress={}/{} life_exhausted={}/{} (independent endpoints)",
            population
                .iter()
                .filter(|p| p["low_progress"] == true)
                .count(),
            n_seres,
            population
                .iter()
                .filter(|p| p["life_exhausted"] == true)
                .count(),
            n_seres
        );

        hist_med.push(r.dist_med);
        // D6: o melhor da geração vira o fantasma da próxima.
        if fantasma_on || escada_ctl.is_some() {
            let best_i = fitness
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            fantasma_atual = fantasma::Fantasma::gravar(&cruzamentos[best_i]);
        }
        // ESCADA: avalia a dose do mundo e sobe/segura/recua degraus (escada.rs).
        if let Some(ctl) = escada_ctl.as_mut() {
            hist_med_escada.push(r.dist_med);
            let promo_pct = r.n_ge1v as f32 / n_seres as f32;
            let recuperou = hist_med_escada.len() >= 15
                && r.dist_med >= hist_med_escada[hist_med_escada.len() - 15] * 0.8;
            let veredito = ctl.avaliar(promo_pct, r.mortality, recuperou);
            match &veredito {
                escada::Veredito::Sobe(d) | escada::Veredito::Rece(d) => match d {
                    escada::Degrau::FuelK(k) => {
                        FUEL_AUTO_K.store(*k, Ordering::Relaxed);
                    }
                    escada::Degrau::Ruido(sig) => {
                        ctx.brain.set_noise(*sig, 0.02);
                    }
                    escada::Degrau::Dano(e) => {
                        DANO_ESCALA.store(*e, Ordering::Relaxed);
                    }
                    escada::Degrau::Clima(mu) => {
                        ctx.phys_gpu.set_clima(*mu, 0.0, 0.0, 0.03);
                    }
                    escada::Degrau::Fantasma => {
                        fantasma_on = true;
                    }
                    // D7 — the ladder turning the scarce resource on/tighter.
                    //
                    // It takes the sector refill away at the same moment, exactly as
                    // the CLI path does: D7 REPLACES the refill, and leaving both on
                    // would put an infinite fuel source next to a scarce one, which is
                    // a pressure that reads as on and behaves as off. Done here rather
                    // than once at start-up because the ladder can reach D7 in the
                    // middle of a battery that began with the refill enabled.
                    escada::Degrau::Resource(m) => {
                        STATIONS_N.store(*m, Ordering::Relaxed);
                        if SETOR_PAGA.load(Ordering::Relaxed) {
                            SETOR_PAGA.store(false, Ordering::Relaxed);
                            REFUEL_LAPS.store(u32::MAX, Ordering::Relaxed);
                            println!(
                                "  [recurso] D7 turned on by the ladder: per-sector refill \
                                 DISABLED, {m} stations become the only source."
                            );
                        }
                        // The being cannot learn a pressure it cannot sense. The ladder
                        // is autonomous, so it turns the body block on rather than
                        // leaving a silent unsensed degree running for 200 generations.
                        if !universe::gpu::perception::BODY_CHANNELS_ON.load(Ordering::Relaxed) {
                            universe::gpu::perception::BODY_CHANNELS_ON
                                .store(true, Ordering::Relaxed);
                            println!(
                                "  [interocepcao] turned on by the ladder together with D7 — \
                                 without the station channels (ch114-116) the rung would be a \
                                 pressure the being cannot feel."
                            );
                        }
                    }
                    // D8 — the ladder tightening the sensory subscription. The frozen
                    // reference `ref_familias` is computed on the first generation the
                    // dose is non-zero (see the block next to the wiring cost).
                    escada::Degrau::SensorCost(dose) => {
                        SENSOR_COST.store(*dose, Ordering::Relaxed);
                    }
                },
                escada::Veredito::Mantem => {}
            }
            println!("{}", ctl.descrever());
        }
        // ── HOSTILE WORLD: telemetry, one line per mechanism per generation ────
        //
        // These lines are the dose canary's readout. The rule they exist to serve:
        // different doses must produce DIFFERENT numbers, including the arm with the
        // mechanism off — a dose that does not differentiate is not weak pressure, it
        // is pressure that is switched off (ESCADA_DIFICULDADE.md, permanent canary).
        // Two silent bugs were caught by exactly this before, and neither showed up in
        // the fitness.
        if !ctx.stations.is_empty() {
            let cap = ctx.station_cfg.capacity.max(1e-6);
            let carga_med: f32 =
                ctx.stations.iter().map(|s| s.charge).sum::<f32>() / ctx.stations.len() as f32;
            // Spread of fuel among the LIVING. Competition should OPEN the spread;
            // an unchanged spread with the stations on says nobody actually lost the
            // dispute. Corpses are excluded — a dead being's tank is frozen at whatever
            // killed it, and averaging that in is the "say who the statistic is about"
            // rule (CLAUDE.md rule 1).
            let vivos: Vec<f64> = states
                .iter()
                .filter(|s| s.life_left > 0.0)
                .map(|s| s.fuel)
                .collect();
            let (med_f, dp_f) = if vivos.is_empty() {
                (0.0, 0.0)
            } else {
                let m = vivos.iter().sum::<f64>() / vivos.len() as f64;
                let v = vivos.iter().map(|f| (f - m) * (f - m)).sum::<f64>() / vivos.len() as f64;
                (m, v.sqrt())
            };
            println!(
                "  [recurso] stations {} (exhausted {}), mean charge {carga_med:.3}/{cap:.2}, \
                 fuel transferred {:.3} u, beings that ran dry {}/{n_seres}, \
                 fuel of the LIVING mean {med_f:.4} sd {dp_f:.4} (n={})",
                ctx.stations.len(),
                ctx.stations_exhausted(),
                ctx.tel_station_fuel,
                ctx.tel_dry_beings,
                vivos.len(),
            );
        }
        if ctx.predator.is_some() {
            // Being-ticks under the drain, not "beings caught": the drain is a ramp, so
            // there is no moment of capture to count, and a threshold invented for the
            // log would report a number the mechanism does not have.
            println!(
                "  [predador] v_cap {:.2} m/s, being-ticks under drain {} ({:.2}% of the {} \
                 being-ticks of the generation), life drained {:.0} ticks (mean {:.1}/being)",
                ctx.predator_params.speed_cap,
                ctx.tel_predator_ticks,
                100.0 * ctx.tel_predator_ticks as f64 / (n_ticks as f64 * n_seres as f64).max(1.0),
                n_ticks * n_seres,
                ctx.tel_predator_life,
                ctx.tel_predator_life / n_seres as f64,
            );
        }
        if !ctx.thermal.is_empty() && ctx.tel_tire_temp_n > 0 {
            let med = ctx.tel_tire_temp_sum / ctx.tel_tire_temp_n as f64;
            // The SWING is the dose canary, not the mean — see `tel_tire_temp_sq`.
            let dp = (ctx.tel_tire_temp_sq / ctx.tel_tire_temp_n as f64 - med * med)
                .max(0.0)
                .sqrt();
            println!(
                "  [pneu] temperature swing sd {dp:.4} ({:.1} C) — THIS is the dose canary; \
                 an sd that does not move with the dose means the model is not integrating.",
                dp * universe::tire_thermal::T_SCALE_CELSIUS as f64,
            );
            println!(
                "  [pneu] mean normalised temperature {med:.3} ({:.0} C) over {} being-ticks; \
                 population mean optimum {:.3}",
                universe::tire_thermal::celsius(med as f32),
                ctx.tel_tire_temp_n,
                ctx.thermal.iter().map(|p| p.optimal).sum::<f32>() / ctx.thermal.len() as f32,
            );
        }
        if !ctx.sensor_cost.is_empty() {
            let m = ctx.sensor_cost.iter().sum::<f32>() / ctx.sensor_cost.len() as f32;
            let lo = ctx.sensor_cost.iter().copied().fold(f32::MAX, f32::min);
            let hi = ctx.sensor_cost.iter().copied().fold(f32::MIN, f32::max);
            println!(
                "  [sensor] tax life/tick: mean {m:.4} min {lo:.4} max {hi:.4} \
                 (spread {:.4} — if it is ~0 the tax became a constant and does not select, \
                 which is how E-REG1 failed)",
                hi - lo,
            );
        }
        hist_lap.push(lap_dist);
        hist_max.push(r.dist_max);

        // ── Salva melhor genoma ────────────────────────────────────────────────
        // Usa gen_abs no nome do arquivo — continua sequência do checkpoint.
        // Melhor pela MESMA grandeza que a seleção usa — senão o genoma salvo e o
        // relatório de reprodutibilidade falam de um campeão que a evolução não escolheu.
        let best_idx = selection_fitness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let save_path = format!("{}/evolve_best_gen{:04}.bin", out_dir, gen_abs);
        let best_physical_slot = best_idx * school_eval_starts;
        if let Err(e) = save_genome(&genomes[best_physical_slot], &save_path) {
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
        if let Some(curriculum) = adaptive.as_mut() {
            let score = if organism_taxi {
                ctx.organism_samples.iter().filter(|s| s.values[68] >= 1.0 && s.values[29] > 0.5).count() as f64 / n_seres.max(1) as f64
            } else { v1 as f64 / n_seres.max(1) as f64 };
            let event = curriculum
                .observe(score)
                .expect("valid training success fraction");
            let record = serde_json::json!({"schema_version":1,"generation":gen_abs,
                "training_seed":seed_gen,"success_fraction":score,
                "frontier_index":curriculum.frontier_index(),"mastered":curriculum.mastered(),
                "event":event.map(|e| format!("{:?}",e.kind)),"source":if organism_taxi { "taxi_delivery_and_survival_training_only" } else { "ordered_lap_crossings_training_only" }});
            std::fs::write(
                format!("{out_dir}/curriculum_gen{gen_abs:04}.json"),
                record.to_string(),
            )
            .expect("write curriculum provenance");
        }
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
        let n_portais = if PORTAL_CURVA.load(Ordering::Relaxed) {
            portais.len() as u32
        } else {
            N_CHECKPOINTS
        };
        if !organism_taxi {
            println!(
                "  [setores] mediana {}/{} ({:.1}% da volta) · melhor {}/{} ({:.1}%) · voltas reais: {} seres ≥1, {} ≥2",
                cp_med, n_portais, 100.0 * cp_med as f32 / n_portais.max(1) as f32,
                cp_max, n_portais, 100.0 * cp_max as f32 / n_portais.max(1) as f32, v1, v2
            );
            let lap_records = lap_tracker.finish();
            let lap_summary = laps::summary(&lap_records);
            println!("  {}", lap_summary.log_line(n_seres));
            let lap_seeds: Vec<u64> = (0..n_seres)
                .map(|i| population_seed_for(pop_seed, i))
                .collect();
            let lap_path = format!("{}/laps_gen{:04}.jsonl", out_dir, gen_abs);
            if let Err(e) = laps::write_jsonl(&lap_path, &lap_records, &lap_seeds) {
                eprintln!(
                    "[gen {:4}] WARNING: could not write {}: {}",
                    gen_abs, lap_path, e
                );
            }
        }

        // ── Painel: fim de geração (E5) ─────────────────────────────────────────
        // Fica AQUI (não logo após `hist_med.push`, mais acima) porque é onde `checkpoint`
        // e `n_portais` (para `setores`) já estão prontos — e `hist_med` já recebeu o push
        // desta geração, então `linear_slope` já enxerga o ponto atual.
        if let Some(p) = &painel {
            let setores_frac: Vec<f32> = (0..n_portais)
                .map(|s| {
                    checkpoint.iter().filter(|&&c| c > s).count() as f32 / n_seres.max(1) as f32
                })
                .collect();
            let slope = if hist_med.len() >= 2 {
                linear_slope(&hist_med).0
            } else {
                0.0
            };
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
            let mut v: Vec<f64> = (0..n_seres)
                .map(|i| {
                    let seg = (states[i].ticks_alive.max(1) as f64) / 64.0;
                    dist_acum[i] / seg
                })
                .collect();
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
        // Achado "UM ser rastejando promove a turma inteira" (`--switch-on-complete 1`):
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
        if !organism_taxi && trocar_ao_completar > 0 && (cond_populacao || cond_veloc) {
            // Currículo de TETO DE MARCHA: a promoção sobe o corpo ANTES de rodar a
            // pista — um degrau por vez. Só quando o teto está no máximo a promoção
            // volta a rodar a pista.
            let teto_atual = CAMBIO_TETO.load(Ordering::Relaxed);
            let teto_max = CAMBIO_TETO_MAX.load(Ordering::Relaxed);
            if CAMBIO_CURRICULO.load(Ordering::Relaxed)
                && CAMBIO_MODO.load(Ordering::Relaxed) > 0
                && teto_atual < teto_max
            {
                CAMBIO_TETO.store(teto_atual + 1, Ordering::Relaxed);
                println!(
                    "  [cambio] população completou a volta ({v1} seres) — teto de marcha {} → {} (pista continua {})",
                    teto_atual,
                    teto_atual + 1,
                    world_seed_atual
                );
            } else {
                idx_curriculo += 1;
                let proxima = pista_da_gen(&pistas, idx_curriculo);
                let via = if cond_populacao {
                    format!(
                        "população (v1={} ≥ {} = max(N={}, 25%={}))",
                        v1, limiar_efetivo, trocar_ao_completar, piso_populacional
                    )
                } else {
                    format!(
                        "velocidade (v1={} ≥ N={} · mediana {:.2}m/s ≥ piso {:.2}m/s)",
                        v1, trocar_ao_completar, vmed, curriculo_veloc_min
                    )
                };
                println!(
                    "  [curriculo] {} seres completaram a volta via {} — pista {} → {}",
                    v1, via, world_seed_atual, proxima
                );
            }
        } else if !organism_taxi && trocar_ao_completar > 0 && v1 > 0 {
            // Visível mesmo quando NÃO promove: sem isto, "por que não trocou" exige
            // reler o código toda vez que alguém completa a volta e a pista não muda.
            println!(
                "  [curriculo] {} seres completaram a volta, mas nem população (precisa {}) \
                 nem velocidade (mediana {:.2}m/s, precisa {:.2}m/s com N={}) bateram o piso \
                 — pista continua {}",
                v1,
                limiar_efetivo,
                vmed,
                curriculo_veloc_min,
                trocar_ao_completar,
                world_seed_atual
            );
        }

        if organism_taxi {
            println!("  [taxi-runtime] generation={gen_abs} elapsed_seconds={t_gen_s:.1}");
        } else {
        println!(
            "gen {:4} | med={:7.0}m ({:.2}V) max={:7.0}m ({:.2}V) | ≥1V={} ≥2V={} low_progress={:.0}% | {:.1}s",
            gen_abs,
            r.dist_med,
            r.dist_med / lap_dist,
            r.dist_max,
            r.dist_max / lap_dist,
            r.n_ge1v,
            r.n_ge2v,
            r.mortality * 100.0,
            t_gen_s,
        );

        }

        let sem_vida = states
            .iter()
            .take(n_seres)
            .filter(|s| s.life_left <= 0.0)
            .count();
        if !organism_taxi && FUEL_ON.load(Ordering::Relaxed) {
            let secos = states
                .iter()
                .take(n_seres)
                .filter(|s| s.fuel <= 0.0)
                .count();
            let mut f: Vec<f64> = states
                .iter()
                .take(n_seres)
                .map(|s| s.fuel.max(0.0))
                .collect();
            f.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  [combustível] secos {}/{} · tanque restante mediano {:.2} · tanque inicial {:.2}",
                secos,
                n_seres,
                f[f.len() / 2],
                FUEL_TANK.load(Ordering::Relaxed)
            );
        }
        // Velocidade média da população. É a grandeza que a vida-ganha ataca: a distância
        // pode subir só por ter mais tempo, mas a velocidade não mente.
        // (`vmed` já foi calculada acima, antes do gate do currículo — reusa aqui.)
        if !organism_taxi {
            let vida_med = states
                .iter()
                .take(n_seres)
                .map(|s| s.ticks_alive as f64)
                .sum::<f64>()
                / n_seres as f64;
            {
                // Hodômetro contra progresso: a lacuna entre os dois É o quanto o ser anda sem
                // avançar na pista. Com fitness de hodômetro essa lacuna era invisível e ele
                // podia ir reto para fora do circuito sem perder nada.
                let mut od: Vec<f64> = dist_acum.clone();
                od.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut pr: Vec<f64> = prog_acum.clone();
                pr.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let (odm, prm) = (od[od.len() / 2], pr[pr.len() / 2]);
                println!(
                    "  [pista] hodômetro {:.0} m · progresso {:.0} m · aproveitamento {:.0}%",
                    odm,
                    prm,
                    if odm > 1.0 { 100.0 * prm / odm } else { 0.0 }
                );
            }
            // Telemetria do portão do piloto: quanto da corrida o CÉREBRO reassumiu.
            // É a medida direta de "onde a fórmula não sabe" — e o número que diz se o
            // domínio destilado cobre a pista nova (ver E-PIL1 em docs/DESTILABILIDADE.md).
            if let Some(pl) = ctx.piloto_formula.as_ref() {
                let tot = pl.decisoes + pl.abstencoes;
                if tot > 0 {
                    println!(
                        "  [piloto] fórmula decidiu {} ticks · cérebro reassumiu {} ({:.1}%)",
                        pl.decisoes,
                        pl.abstencoes,
                        100.0 * pl.abstencoes as f64 / tot as f64
                    );
                    let (med, mx, q) = pl.estat_seqs();
                    println!(
                        "  [piloto] sequências de abstenção: {q} blocos · mediana {med:.0} ticks · maior {mx}"
                    );
                }
            }
            if NEUROMOD_REGIOES.load(Ordering::Relaxed) {
                println!(
                    "  [neuromod] gates ser0 [visão,parede,corpo,assoc] = [{:.2},{:.2},{:.2},{:.2}] · reserva [{:.2},{:.2},{:.2},{:.2}]",
                    ctx.ultimo_gate[0],
                    ctx.ultimo_gate[1],
                    ctx.ultimo_gate[2],
                    ctx.ultimo_gate[3],
                    states[0].nm_reserva[0],
                    states[0].nm_reserva[1],
                    states[0].nm_reserva[2],
                    states[0].nm_reserva[3]
                );
            }
            println!(
                "  [veloc] mediana {:.2} m/s ({:.1} km/h) | pico {:.1} m/s | marcha máx {} | vida média {:.0}% da geração",
                vmed,
                vmed * 3.6,
                VEL_PICO.load(Ordering::Relaxed),
                MARCHA_PICO.load(Ordering::Relaxed),
                100.0 * vida_med / n_ticks as f64
            );
            VEL_PICO.store(0.0, Ordering::Relaxed);
            MARCHA_PICO.store(0, Ordering::Relaxed);
        }
        if organism_taxi {
            println!("  [organism-contact] energy_j={:.3} impact_ticks={} dead={}/{}",
                ctx.organism_samples.iter().map(|s| s.values[100] as f64).sum::<f64>(),
                ctx.organism_samples.iter().map(|s| s.values[101] as u64).sum::<u64>(), sem_vida,n_seres);
        } else {
        println!(
            "  [parede] {:.1}% dos ticks colado (<{:.0} m) | sem vida: {}/{} | custo={:.0} m/s",
            100.0 * wall_frac,
            WALL_NEAR_M,
            sem_vida,
            n_seres,
            wall_cost
        );
        if impact_cost > 0.0 {
            let total_energy_j: f64 = states.iter().map(|state| state.collision_energy_j).sum();
            let total_impacts: u64 = states
                .iter()
                .map(|state| u64::from(state.impact_count))
                .sum();
            println!(
                "  [impact] {:.1} kJ/being · {} impact steps · cost={:.3} m/kJ",
                total_energy_j / 1000.0 / n_seres as f64,
                total_impacts,
                impact_cost,
            );
        }
        let hole_hits: u64 = states.iter().map(|state| u64::from(state.hole_hits)).sum();
        println!(
            "  [survival] hole contact episodes={} · fuel dry beings={}",
            hole_hits, ctx.tel_dry_beings
        );

        }

        // ── O lento merece existir? ────────────────────────────────────────────
        // Razão de participação das três saídas: perto de 3 = três decisões distintas;
        // perto de 1 = um canal usando três nomes, e crescer o órgão seria só espaço.
        if !ctx.slow_samples.is_empty() {
            let samples = std::mem::take(&mut ctx.slow_samples);
            let pr = universe::math::linalg::participation_ratio(&samples);
            let amp: Vec<f32> = (0..3)
                .map(|k| {
                    let mx = samples.iter().map(|r| r[k]).fold(f32::MIN, f32::max);
                    let mn = samples.iter().map(|r| r[k]).fold(f32::MAX, f32::min);
                    mx - mn
                })
                .collect();
            match pr {
                Some(v) => println!(
                    "  [lento] PR={:.2}/3 ({:.0}% da capacidade) | amplitude {:.3} {:.3} {:.3} | n={}",
                    v,
                    100.0 * v / 3.0,
                    amp[0],
                    amp[1],
                    amp[2],
                    samples.len()
                ),
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

            let (l1_stats, l1_alarms) =
                collect_activation_stats("L1", &l1, Some(l1_w), L1_NEURONS, None);
            let (l2_stats, l2_alarms) =
                collect_activation_stats("L2", &l2, None, L2_NEURONS, Some(&policy));

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
                let ganho_fitness_geracao: Option<f32> =
                    elite_fitness_anterior
                        .map(|ant| (selection_fitness[0] - ant).max(0.0) as f32);
                let disp_fitness: f32 = {
                    let vals: Vec<f32> = fitness.iter().map(|f| *f as f32).collect();
                    universe::math::stats::std_dev(&vals)
                };
                for (nome, amostras) in [("L1", &l1), ("L2", &l2)] {
                    if amostras.len() < 2 {
                        continue;
                    }
                    let e = ctx::esparsidade(nome, amostras, LIMIAR);
                    let p = ctx::persistencia(nome, amostras, LIMIAR, 37);
                    let q = ctx::quente_frio(nome, amostras, LIMIAR);
                    println!(
                        "  [contextual:{}] limiar={:.2} · ativos {:.1}%±{:.1} · saturado {:.1}%",
                        nome,
                        LIMIAR,
                        e.fracao_ativa_media * 100.0,
                        e.fracao_ativa_sigma * 100.0,
                        e.saturacao_media * 100.0
                    );
                    println!(
                        "  [contextual:{}] persistencia {:.1}% · baseline {:.1}% · ganho {:+.1} pts · churn {:.1}%",
                        nome,
                        p.persistencia * 100.0,
                        p.baseline_aleatorio * 100.0,
                        p.ganho_temporal * 100.0,
                        p.churn * 100.0
                    );
                    println!(
                        "  [contextual:{}] {} neuronios cobrem 80% das ativacoes · {} nunca · {} sempre",
                        nome, q.neuronios_para_80pct, q.nunca_ativos, q.sempre_ativos
                    );
                    // Armadilha nº 1 do protocolo: `tanh` nao zera, entao "ativo" e uma
                    // escolha de limiar. A curva mostra se a conclusao sobrevive a ela.
                    let sens = ctx::sensibilidade_ao_limiar(amostras, &[0.01, 0.05, 0.2, 0.5, 0.9]);
                    let curva: Vec<String> = sens
                        .iter()
                        .map(|(l, f)| format!("{l:.2}→{:.0}%", f * 100.0))
                        .collect();
                    println!(
                        "  [contextual:{}] sensibilidade ao limiar: {}",
                        nome,
                        curva.join(" · ")
                    );

                    // E3 — co-ativacao. E a base EMPIRICA para dividir o cerebro em
                    // regioes: os grupos tem de sair de neuronios que disparam juntos,
                    // nao de caixas desenhadas a priori. Excesso ~0 significa que cada
                    // neuronio decide sozinho — sem estrutura de grupo, nao ha regiao a
                    // separar (foi o que pesos aleatorios deram no protocolo: -0,1%).
                    let co = ctx::coativacao(nome, amostras, LIMIAR, 5);
                    println!(
                        "  [contextual:{}] co-ativacao {:.3} vs independente {:.3} · excesso {:+.3}",
                        nome, co.coativacao_media, co.esperado_independente, co.excesso
                    );
                    if !co.pares_fortes.is_empty() {
                        let ps: Vec<String> = co
                            .pares_fortes
                            .iter()
                            .map(|(i, j, e)| format!("({i},{j}){e:+.2}"))
                            .collect();
                        println!(
                            "  [contextual:{}] pares mais acoplados: {}",
                            nome,
                            ps.join(" ")
                        );
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
                            nome,
                            pr,
                            n_neur,
                            100.0 * pr / n_neur as f32
                        );

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
                            amostras,
                            n_neur,
                            0.80,
                            ganho,
                            disp_fitness,
                            n_seres,
                        );
                        let (rotulo, acao) = match v {
                            universe::brain_capacity::Verdict::Grow => (
                                "CRESCER",
                                "capacidade saturada e ganho acima do ruido de busca",
                            ),
                            universe::brain_capacity::Verdict::WastedSpace => (
                                "nao crescer",
                                "ha dimensoes ociosas — crescer desperdicaria",
                            ),
                            universe::brain_capacity::Verdict::GainNotReal => (
                                "nao crescer",
                                "saturada, mas o ganho nao passa do limiar do Garimpo",
                            ),
                            universe::brain_capacity::Verdict::InsufficientData => (
                                "indefinido",
                                "amostras insuficientes para estimar a covariancia",
                            ),
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

                for (nome, amostras, contexto) in [("L1", &l1, &regiao_l1), ("L2", &l2, &regiao_l2)]
                {
                    match regioes::analisar(
                        nome,
                        amostras,
                        contexto,
                        LIMIAR_CURVA,
                        LIMIAR_ESPECIALIZACAO,
                        LIMIAR_ATIVACAO_COESAO,
                    ) {
                        Some(r) => {
                            println!(
                                "  [regioes:{}] limiar_curva={:.3} 1/m · amostras reta={} curva={}",
                                nome, r.limiar_curva, r.n_reta, r.n_curva
                            );
                            println!(
                                "  [regioes:{}] especializacao >= {:.2}: {}/{} neuronios · idx {:?}",
                                nome,
                                r.limiar_especializacao,
                                r.indices_especializados.len(),
                                r.por_neuronio.len(),
                                r.indices_especializados
                            );
                            let mut ord = r.por_neuronio.clone();
                            ord.sort_by(|a, b| {
                                b.indice_especializacao
                                    .partial_cmp(&a.indice_especializacao)
                                    .unwrap()
                            });
                            let top: Vec<String> = ord.iter().take(5).map(|e| format!(
                                "#{}(idx={:.2} corr_curva={:+.2} corr_vel={:+.2} corr_parede={:+.2})",
                                e.indice, e.indice_especializacao, e.correlacao_curvatura,
                                e.correlacao_velocidade, e.correlacao_dist_parede)).collect();
                            println!("  [regioes:{}] top especializados: {}", nome, top.join(" "));
                            match &r.coesao {
                                Some(co) => println!(
                                    "  [regioes:{}] coesao dos especializados: co-ativacao {:.3} vs independente {:.3} · excesso {:+.3}",
                                    nome, co.coativacao_media, co.esperado_independente, co.excesso
                                ),
                                None => println!(
                                    "  [regioes:{}] coesao: N/A (menos de 2 especializados ou menos de 2 amostras de curva)",
                                    nome
                                ),
                            }
                        }
                        None => println!(
                            "  [regioes:{}] sem dados (amostras={} contexto={} — desalinhado ou vazio)",
                            nome,
                            amostras.len(),
                            contexto.len()
                        ),
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
                let dead_pct: f32 =
                    stats.neurons.iter().map(|s| s.dead_pct).sum::<f32>() / n_neurons as f32;
                let saturad_pct: f32 =
                    stats.neurons.iter().map(|s| s.saturated_pct).sum::<f32>() / n_neurons as f32;
                let snr_pct: f32 =
                    stats.neurons.iter().map(|s| s.snr_i8).sum::<f32>() / n_neurons as f32;
                println!(
                    "  [ativacao:{}] amostras={} dead_pct={:.1}% saturad_pct={:.1}% \
                     raio_espectral={} rank_efetivo={} alarmes={}",
                    stats.layer_name,
                    stats.n_samples,
                    dead_pct,
                    saturad_pct,
                    stats
                        .spectral_radius
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "n/a".into()),
                    stats
                        .effective_rank
                        .map(|k| format!("{k}/{n_neurons}"))
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
                        live: ultima_amostra_camada
                            .get(&stats.layer_name)
                            .cloned()
                            .unwrap_or_default(),
                        mean_per_neuron: stats.neurons.iter().map(|n| n.mean).collect(),
                    });
                    for a in alarms {
                        painel_alarms.push(format!(
                            "{}: {:?} valor={:.2} limiar={:.2}",
                            a.layer, a.kind, a.value, a.threshold
                        ));
                    }
                }
            }
            if let Some(pe) = l2_stats.policy_entropy {
                println!(
                    "  [ativacao:L2] entropia_politica={:.3} nats (limiar {:.2}) · ser 0 = campeão herdado",
                    pe,
                    universe::math::thresholds::POLICY_ENTROPY_MIN,
                );
            }
            if !nan_seres.is_empty() {
                println!(
                    "  [ativacao] ALARME: NaN/Inf detectado no estado L1 de {} ser(es) nesta geração",
                    nan_seres.len(),
                );
                if painel.is_some() {
                    painel_alarms.push(format!(
                        "NaN/Inf detectado no estado L1 de {} ser(es) nesta geração",
                        nan_seres.len()
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
        //
        // Reads `effective_tau()`, not `tau`: under `--tau-bank log-fixed`/`spectral`
        // the raw genes are no longer what the GPU sees, and printing them would answer
        // a question nobody asked — the dispersion that matters is the one the CfC runs
        // with. Under `livre` the two are the same vector, so the line is unchanged.
        {
            let campeao = &genomes[best_idx];
            let mut taus: Vec<f32> = campeao.effective_tau().iter().map(|&d| 1.0 / d).collect();
            taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p10 = taus[taus.len() / 10];
            let p50 = taus[taus.len() / 2];
            let p90 = taus[taus.len() * 9 / 10];
            // Dispersão da população inteira, não só do campeão: um elite com τ espalhado
            // dentro de uma população uniforme seria sorte, não seleção.
            let disp_pop: f32 = genomes
                .iter()
                .map(|g| {
                    let mut t: Vec<f32> = g.effective_tau().iter().map(|&d| 1.0 / d).collect();
                    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    t[t.len() * 9 / 10] / t[t.len() / 10].max(1e-6)
                })
                .sum::<f32>()
                / genomes.len() as f32;
            println!(
                "  [tau] campeão τ p10 {:.3}s · mediana {:.3}s · p90 {:.3}s (espalha {:.1}×) | população {:.1}×",
                p10,
                p50,
                p90,
                p90 / p10.max(1e-6),
                disp_pop
            );
            // The mode goes on its own line instead of into the format string above, so
            // `free` prints byte-identically to every run that predates `--tau-bank`
            // (the exact format quoted in docs/CHAMPION_RELOAD.md:50-51 included).
            let bank_mode = tau_bank_mode();
            if bank_mode != universe::tau_bank::TauBankMode::Free {
                println!(
                    "  [tau] the numbers above are the EFFECTIVE bank of --tau-bank {} — {}",
                    bank_mode.as_flag(),
                    bank_mode.describe()
                );
            }
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
                    &mut ctx,
                    &genomes,
                    evolving_body,
                    n_seres,
                    n_ticks,
                    track_width,
                    same_start,
                    &modo_fitness,
                    wall_cost,
                    impact_cost,
                    ho_seed,
                    &kan_knots_global,
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
                ctx.reload_world(&world);

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
                let razao = if treino_med > 1.0 {
                    held_med / treino_med
                } else {
                    f64::NAN
                };
                println!(
                    "  [generalizacao] gen={:4} treino={:.3}{} held={:.3}{} razao={:.3}  \
                     (pista treino=seed {} · held-out=seed {})",
                    gen_abs, treino_med, if organism_taxi {" score"} else {"m"}, held_med, if organism_taxi {" score"} else {"m"}, razao, seed_gen, ho_seed,
                );
            }
        }

        // O genoma que vai para o slot 0 da próxima geração é o melhor DESTA.
        elite_fitness_anterior = Some(
            selection_fitness
                .iter()
                .cloned()
                .fold(f64::MIN, f64::max),
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
            let selection_genomes = school_batch::representatives(&genomes, school_eval_starts);
            genomes = evolve(
                &selection_genomes,
                &selection_fitness,
                &evo_params,
                gen_abs,
                pop_seed,
            );
            if organism_taxi {
                universe::gpu::body::mutate_organism_population(
                    &mut genomes,
                    evo_params.n_elite,
                    gen_abs,
                    pop_seed,
                );
            }
            if evolving_body {
                universe::gpu::body::mutate_population(
                    &mut genomes,
                    evo_params.n_elite,
                    gen_abs,
                    pop_seed,
                );
            }
            let mutation_flags = (if evolve_learning { 1 } else { 0 }) | (lifetime_flags & 6);
            if mutation_flags != 0 {
                for (i, genome) in genomes.iter_mut().enumerate().skip(evo_params.n_elite) {
                    let seed = pop_seed
                        ^ (gen_abs as u64).wrapping_mul(0x9e3779b97f4a7c15)
                        ^ (i as u64).wrapping_mul(0xbf58476d1ce4e5b9);
                    genome.lifetime = genome.lifetime.mutated(seed, mutation_flags);
                }
            }
            if school_eval_starts > 1 {
                genomes = school_batch::expand_replicas(&genomes, school_eval_starts);
            }
        }
    }

    let total_s = t_total.elapsed().as_secs_f64();
    if organism_taxi {
        println!("=== ORGANISM GPU RUN COMPLETE: {n_gens} generations, {n_seres} beings, {total_s:.1}s ===");
        println!("Task outcomes and resource budgets: {out_dir}/organism_result_gen*.json");
        println!("Execution success is not evidence of learned delivery or generalization.");
        return;
    }

    println!();
    if !elite_reprod.is_empty() {
        let validos: Vec<f64> = elite_reprod
            .iter()
            .cloned()
            .filter(|v| v.is_finite())
            .collect();
        if !validos.is_empty() {
            let media = validos.iter().sum::<f64>() / validos.len() as f64;
            let mut ord = validos.clone();
            ord.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mediana = ord[ord.len() / 2];
            let piores = ord.iter().filter(|&&v| v < 0.5).count();
            println!();
            println!("=== REPRODUTIBILIDADE DO CAMPEÃO ===");
            println!("  o MESMO genoma, reavaliado na geração seguinte:");
            println!(
                "  média {:.0}% · mediana {:.0}% do resultado anterior · n={}",
                100.0 * media,
                100.0 * mediana,
                validos.len()
            );
            println!(
                "  perdeu mais da METADE em {}/{} gerações",
                piores,
                validos.len()
            );
            println!("  (100% = fitness é do genoma; muito abaixo = seleção premiou sorte)");
        }
    }

    println!("=== RESULTADO FINAL ({:.1}s total) ===", total_s);
    println!(
        "  gerações: {} · seres: {} · ticks/gen: {}",
        n_gens, n_seres, n_ticks
    );
    if let Some(fly) = ctx.fly_spiking.as_ref() {
        let traffic = fly.traffic();
        let ticks = traffic.ticks.max(1);
        println!(
            "  [fly-spiking][tráfego] CPU→GPU total {:.2} MiB ({} chamadas) · setup/reset {:.2} MiB",
            traffic.cpu_to_gpu_bytes as f64 / (1024.0 * 1024.0),
            traffic.cpu_to_gpu_calls,
            traffic
                .cpu_to_gpu_bytes
                .saturating_sub(traffic.tick_cpu_to_gpu_bytes) as f64
                / (1024.0 * 1024.0),
        );
        println!(
            "  [fly-spiking][tráfego] CPU→GPU por tick {:.1} B ({:.2} uploads/tick) · GPU→CPU total {:.2} MiB ({} readbacks)",
            traffic.tick_cpu_to_gpu_bytes as f64 / ticks as f64,
            traffic.tick_cpu_to_gpu_calls as f64 / ticks as f64,
            traffic.gpu_to_cpu_bytes as f64 / (1024.0 * 1024.0),
            traffic.gpu_to_cpu_calls,
        );
        println!(
            "  [fly-spiking][tráfego] GPU→CPU por tick {:.1} B ({:.2} mapas/tick) · cópias internas total {:.2} MiB, por tick {:.1} B",
            traffic.tick_gpu_to_cpu_bytes as f64 / ticks as f64,
            traffic.tick_gpu_to_cpu_calls as f64 / ticks as f64,
            traffic.gpu_internal_copy_bytes as f64 / (1024.0 * 1024.0),
            traffic.tick_gpu_internal_copy_bytes as f64 / ticks as f64,
        );
    }
    println!();

    // ── Última geração ─────────────────────────────────────────────────────────
    if let Some(last) = hist_results.last() {
        println!("Última geração (gen {:4}):", last.generation);
        println!(
            "  mediana: {:.0}m ({:.3} voltas)",
            last.dist_med,
            last.dist_med / lap_dist
        );
        println!(
            "  máximo:  {:.0}m ({:.3} voltas)",
            last.dist_max,
            last.dist_max / lap_dist
        );
        println!(
            "  ≥1 volta: {}/{}  ≥2 voltas: {}/{}",
            last.n_ge1v, n_seres, last.n_ge2v, n_seres
        );
        println!("  progress below 10 m: {:.1}%", last.mortality * 100.0);
        println!();
    }

    // ── Inclinação da MEDIANA com teste de janela encolhida ───────────────────
    println!("=== TRAJETÓRIA DA MEDIANA (OLS) ===");
    if hist_med.len() >= 2 {
        let (slope, stderr) = linear_slope(&hist_med);
        let sigma_ratio = if stderr > 1e-10 { slope / stderr } else { 0.0 };
        println!(
            "  Todas {} gens: slope={:+.2}m/gen  σ={:.2}  razão={:.2}σ",
            hist_med.len(),
            slope,
            stderr,
            sigma_ratio
        );

        // Teste de janela encolhida: remove as últimas 1, 2, 3 gens e recalcula.
        // Se a inclinação some ao remover poucos pontos, era artefato de janela.
        for remover in 1..=3.min(hist_med.len().saturating_sub(2)) {
            let n_restante = hist_med.len() - remover;
            let (s2, e2) = linear_slope(&hist_med[..n_restante]);
            let sr2 = if e2 > 1e-10 { s2 / e2 } else { 0.0 };
            println!(
                "  -{}gen ({} gens): slope={:+.2}m/gen  σ={:.2}  razão={:.2}σ",
                remover, n_restante, s2, e2, sr2
            );
        }
    } else {
        println!("  Dados insuficientes (< 2 gerações).");
    }

    println!();
    println!("=== TRAJETÓRIA DO MÁXIMO ===");
    if hist_max.len() >= 2 {
        let (slope, stderr) = linear_slope(&hist_max);
        let sigma_ratio = if stderr > 1e-10 { slope / stderr } else { 0.0 };
        println!(
            "  Todas {} gens: slope={:+.2}m/gen  σ={:.2}  razão={:.2}σ",
            hist_max.len(),
            slope,
            stderr,
            sigma_ratio
        );
    }

    // ── Tabela completa das medianas ───────────────────────────────────────────
    println!();
    println!("=== TABELA COMPLETA (gen | mediana metros | voltas) ===");
    for (i, (&med, &max)) in hist_med.iter().zip(hist_max.iter()).enumerate() {
        let ld = hist_lap.get(i).copied().unwrap_or(lap_dist);
        println!(
            "  gen {:4}: med={:7.0}m ({:.3}V) max={:7.0}m ({:.3}V)",
            gen_offset + i,
            med,
            med / ld,
            max,
            max / ld
        );
    }

    println!();

    // Imprime relatório da auditoria se ela foi ativada.
    if AUDITAR.load(Ordering::Relaxed) || audit::n_ticks_acumulados() > 0 {
        audit::auditoria_relatorio();
    }

    println!("=== FIM gpu_evolve ===");
}
