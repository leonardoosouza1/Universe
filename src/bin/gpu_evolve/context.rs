//! GpuCtx: contexto GPU do laço de evolução — retina + percepção + cérebro rápido +
//! cérebro lento + física, tudo num único device wgpu compartilhado.
//!
//! Espelha (estrutura, não código compartilhado) o `GpuLoop` de gpu_loop.rs — a
//! diferença real está em `tick()`: SEM muletas motoras (escape_ticks, curvature
//! governor). O cérebro decide tudo; o piloto fantasma/sensorial (pilots.rs) só entra
//! por flag de diagnóstico.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use wgpu::util::DeviceExt;

use universe::config::iara::vision::{RETINA_ROWS, RETINA_TOTAL};
use universe::engine::SovereignEngine;
use universe::gpu::brain::BrainGpu;
use universe::gpu::slow_brain::SlowBrainGpu;
use universe::gpu::perception::{PerceptionGpu, PosePerception};
use universe::gpu::physics::{PhysicsGpu, PhysicsStateGpu};
use universe::gpu::retina_digest::RetinaDigestGpu;
use universe::gpu::retina_types::{RetinaParamsGpu, PoseRaw};
use universe::gpu::core::{uniform_entry, storage_entry};
use universe::analysis::regioes::{CurvaturaPista, ContextoAmostra};
use universe::world::VoxelWorld;
use universe::simulation::racing_line::RacingLine;
use universe::visao::pitch;
use universe::perception::channels::Channel;
use universe::kan::KAN_OUT_KNOTS;

use crate::statics::*;
use crate::types::BeingState;
use crate::audit::auditoria_acumular;

pub struct GpuCtx {
    pub device:  Arc<wgpu::Device>,
    pub queue:   Arc<wgpu::Queue>,
    pub pipeline_retina: wgpu::ComputePipeline,
    pub layout_retina:   wgpu::BindGroupLayout,
    pub buf_grid:        wgpu::Buffer,
    pub buf_poses_retina: wgpu::Buffer,
    pub buf_retina_out:   wgpu::Buffer,
    pub grid_size:   u32,
    pub cell_size:   f32,
    pub percept:     PerceptionGpu,
    pub brain:       BrainGpu,
    /// LABORATÓRIO DE OBSERVAÇÃO (`--observar`): destino da gravação por tick do
    /// ser 0 — ver o bloco de captura em `tick()` para o layout (394 f32/linha).
    pub obs_writer:  Option<std::io::BufWriter<std::fs::File>>,
    /// E-REG1: custo de fiação por ser (constante durante a geração — pesos não
    /// mudam no meio). Preenchido pelo main a cada aplicação de genomas; drenado
    /// da vida por tick quando CUSTO_FIACAO > 0. Vazio = sem custo.
    pub fiacao_custo: Vec<f32>,
    /// PILOTO POR FÓRMULA: quando presente, o ser 0 é dirigido pelo controlador
    /// ridge (X = debug do cérebro pós-KAN/CfC, mesma fonte do --observar).
    pub piloto_formula: Option<crate::piloto_linear::PilotoLinear>,
    /// M4: pesos do maestro por ser (n×16), preenchidos pelo main por geração.
    pub gates_w: Vec<f32>,
    /// M4: último gate aplicado ao ser 0 (telemetria [neuromod] do main).
    pub ultimo_gate: [f32; 4],
    /// Metade LENTA do cérebro (System 2): CfC-64 → Dense → Dense, 1 tick a cada 8.
    /// Estava ausente de todo o caminho GPU — `grep slow src/gpu/` voltava vazio — o que
    /// significa que "portar o cérebro para a GPU" tinha na verdade portado só a metade
    /// rápida. As três saídas modulam agressão, teto de acelerador e viés de volante.
    pub slow:        SlowBrainGpu,
    /// Amostras da saída do lento ao longo da geração, para MEDIR se ele merece existir.
    /// Uma linha por amostra, três colunas — alimenta `brain_capacity::participation_ratio`.
    pub slow_samples: Vec<Vec<f32>>,
    /// Última intenção neural, reusada nos ticks em que o cérebro não dispara.
    pub last_intents: Vec<[f32; 96]>,
    /// Ticks de física por decisão neural. 1 = decide todo tick (64 Hz).
    ///
    /// É a "velocidade de simulação" pelo lado do cérebro. O `motor.rs` tinha esse
    /// mecanismo adaptativo pelo cortisol e ele foi DESLIGADO com a justificativa
    /// registrada em motor.rs:1717 — "skip_n=2 em gen>=10 reduzia brain freq pra 30Hz e
    /// degradava DNA reload". Ou seja: já se sabia que existe um ótimo interior, e a
    /// resposta foi travar em 1 em vez de medir onde ele fica. Aqui vira parâmetro.
    pub brain_skip: u32,
    pub brain_tick: u32,
    pub racing_line: RacingLine,
    pub grid_cpu:    Vec<u32>,
    pub grid_size_cpu: u32,
    pub cell_size_cpu: f32,
    pub phys_gpu:    PhysicsGpu,
    /// CfC de percepção + min_wall_ahead + edge_row6/7 — kernel `retina_digest.wgsl`.
    /// Substitui o antigo readback de 340 KB/tick (retina inteira) + laço Rust
    /// (`percept_cfc: Vec<RetinaPerception>` rodando na CPU); ver doc do módulo
    /// `gpu::retina_digest` para o antes/depois completo.
    pub retina_digest: RetinaDigestGpu,
    pub capacity:    usize,

    // ── Amostras de ativação (E4, `--ativacao`) ─────────────────────────────
    // Todas bounded por geração: `main.rs` faz `std::mem::take` no relatório de
    // fim-de-geração (mesmo padrão de `slow_samples`) — sem isso acumulariam sem
    // limite numa bateria longa (risco R6 do REFACTOR_PLAN.md).
    /// Estado CfC de L1 (96 valores), uma linha por (ser, tick amostrado).
    pub l1_samples: Vec<Vec<f32>>,
    /// `final_intent[0..48]` (saída pós-L2), uma linha por (ser, tick amostrado).
    /// Reaproveita o readback de `intents` que já acontece todo tick — custo extra
    /// zero além do `l1_samples` (ver comentário no ponto de amostragem em `tick()`).
    pub l2_samples: Vec<Vec<f32>>,
    /// (throttle, steer, brake) do "campeão" (ser 0 = elite herdado da geração
    /// anterior sem mutação — mesmo raciocínio do bloco `[elite]` em main.rs),
    /// amostrado no mesmo passo — usado para entropia de política.
    pub policy_samples: Vec<[f32; 3]>,
    /// Índices de ser com NaN/Inf detectado no estado L1 nesta geração (guarda de
    /// runtime, item 5 do E4). Populado em `tick()`, consumido/relatado em `main.rs`.
    pub nan_seres: std::collections::HashSet<usize>,

    // ── Regiões da pista × ativação (`--regioes`, src/analysis/regioes.rs) ─────
    // Nunca cruzávamos ativação com POSIÇÃO. Estes dois vetores andam em LOCKSTEP
    // com `l1_samples`/`l2_samples` — empilhados na MESMA iteração do mesmo loop,
    // então `regiao_l1[i]` é o contexto físico (curvatura/velocidade/dist_parede)
    // exato do (ser, tick) que produziu `l1_samples[i]`. É o que garante o
    // alinhamento por índice que `analysis::regioes::analisar` exige sem precisar
    // carregar tick/ser junto — mais barato e sem acoplar os dois módulos.
    /// Contexto alinhado a `l1_samples` (mesmo tamanho sempre, mesmo `continue` em NaN).
    pub regiao_l1: Vec<ContextoAmostra>,
    /// Contexto alinhado a `l2_samples` (mesmo tamanho sempre — L2 nunca pula por NaN).
    pub regiao_l2: Vec<ContextoAmostra>,
    /// κ(x,z) da pista atual — recalculado em `new()` e em `reload_world()` (rotação
    /// de pistas, `--pistas`/`--pista-held-out`) para nunca medir curvatura da pista
    /// errada depois de uma troca.
    track_curvatura: CurvaturaPista,
}

/// Contexto físico (curvatura/velocidade/dist_parede) de UM ser neste tick, para
/// pareamento 1:1 com uma amostra de ativação (`--regioes`). Função livre (não método,
/// não closure) de propósito: evita qualquer disputa de borrow com `self` no ponto de
/// chamada, que já mistura leitura de `states`/`phys_states` com `self.l1_samples.push`
/// no mesmo bloco — ver `tick()`.
fn contexto_regiao(track: &CurvaturaPista, p: &PhysicsStateGpu, dist_parede: f32) -> ContextoAmostra {
    let curvatura = track.curvatura_em(p.pos[0], p.pos[2]);
    let velocidade = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
    ContextoAmostra { curvatura, velocidade, dist_parede }
}

impl GpuCtx {
    pub fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
        let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
        if std::path::Path::new(icd_radeon).exists()
            && std::env::var("VK_ICD_FILENAMES").is_err()
        {
            unsafe { std::env::set_var("VK_ICD_FILENAMES", icd_radeon); }
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))?;
        let name = adapter.get_info().name.clone();
        let mut limits = wgpu::Limits::downlevel_defaults();
        // Cérebro usa 10 storage buffers: bindings 1-9 mais o `buf_tau` (banco de τ).
        //
        // O teto é de configuração, não de silício — `downlevel_defaults()` pede 4 para
        // caber em GL antigo, e a RX 6750 XT anuncia bem mais. Pedir 12 dá folga para os
        // próximos dois buffers sem voltar aqui a cada binding novo; se algum adaptador
        // não suportar, `request_device` falha na abertura em vez de corromper dado.
        // M4 (buf_gates, binding 13) estourou os 12 — subido para 16 (RX 6750 XT
        // anuncia bem mais; medido: 12 era teto de config, não de silício).
        limits.max_storage_buffers_per_shader_stage = 16;
        let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("iara-evolve"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )).ok()?;
        let device = Arc::new(raw_device);
        let queue  = Arc::new(raw_queue);
        let cap = capacity.max(1);

        // COPY_DST: permite `reload_world()` trocar de pista via `write_buffer` sem
        // recriar buffer/bind-group/pipeline (P1, rotação de pistas — ver o mesmo
        // raciocínio em `gpu::perception::PerceptionGpu::reload_world`).
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("evolve-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let buf_poses_retina = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("evolve-poses-retina"),
            size:               (cap * std::mem::size_of::<PoseRaw>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_retina_out = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("evolve-retina-out"),
            size:               (cap * RETINA_TOTAL * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("evolve-retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../gpu/retina.wgsl"))),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("evolve-retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pl_retina = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_retina],
            push_constant_ranges: &[],
        });
        let pipeline_retina = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("evolve-retina-pipeline"),
            layout:              Some(&pl_retina),
            module:              &shader_retina,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        let percept = PerceptionGpu::new(&device, &queue, world, cap)?;

        let brain = BrainGpu::new(Arc::clone(&device), Arc::clone(&queue), name.clone(), cap);
        let slow = SlowBrainGpu::new_shared(Arc::clone(&device), Arc::clone(&queue), cap);

        let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
        let phys_gpu = PhysicsGpu::new_shared(
            Arc::clone(&device), Arc::clone(&queue), name.clone(),
            &grid_u8, &world.height_map, world.size, world.cell_size, cap)?;

        let retina_digest = RetinaDigestGpu::new(&device, &queue, cap);

        // κ(x,z) da pista inicial — mesma fonte que `reload_world` usa depois.
        let track_curvatura = {
            let wps: Vec<(f32, f32)> = world.waypoints_world.iter().map(|p| (p.x, p.y)).collect();
            CurvaturaPista::medir(&wps)
        };

        eprintln!("[gpu_evolve] Adapter: {} · cap={}", name, cap);

        Some(Self {
            device, queue,
            pipeline_retina, layout_retina,
            buf_grid, buf_poses_retina, buf_retina_out,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            percept, brain, slow, obs_writer: None, fiacao_custo: Vec::new(),
            piloto_formula: None,
            gates_w: Vec::new(), ultimo_gate: [1.0; 4],
            slow_samples: Vec::new(),
            last_intents: Vec::new(),
            brain_skip: 1,
            brain_tick: 0,
            racing_line: world.racing_line.clone(),
            grid_cpu: grid_u32,
            grid_size_cpu: world.size as u32,
            cell_size_cpu: world.cell_size,
            phys_gpu,
            retina_digest,
            capacity: cap,
            l1_samples: Vec::new(),
            l2_samples: Vec::new(),
            policy_samples: Vec::new(),
            nan_seres: std::collections::HashSet::new(),
            regiao_l1: Vec::new(),
            regiao_l2: Vec::new(),
            track_curvatura,
        })
    }

    /// Carrega pesos dos engines na GPU.
    pub fn upload_engines(&self, engines: &[SovereignEngine]) {
        let refs: Vec<&SovereignEngine> = engines.iter().take(self.capacity).collect();
        self.brain.upload_weights(&refs);
        // Pesos do lento na mesma ordem de `SlowBrain::all_weights()`, para que a
        // evolução mute um vetor só e o kernel leia o layout que já espera.
        let mut slow_w: Vec<f32> = Vec::with_capacity(refs.len() * 1091);
        for e in &refs { slow_w.extend_from_slice(&e.slow_brain.all_weights()); }
        self.slow.upload_weights(&slow_w, refs.len());
    }

    /// Executa um tick do laço GPU.
    /// SEM muletas: sem escape_ticks, sem throttle forçado, sem curvature governor.
    /// O cérebro decide tudo.
    pub fn tick(
        &mut self,
        states:  &mut Vec<BeingState>,
        n:       usize,
        dt:      f32,
        tick_idx: usize,
        kan_knots: &[[f32; KAN_OUT_KNOTS]; 3],
    ) -> bool {
        if n == 0 || n > self.capacity { return false; }

        // Guarda de runtime (E4, REFACTOR_PLAN.md §E4 "NaN-check"): detecta posição
        // não-finita ANTES de submeter à GPU. Só em debug — em release (`cfg!` avaliado
        // em compile-time, custo zero no binário de produção) o guard real fica nas
        // amostras de ativação abaixo, que rodam mesmo com `--release` mas só quando
        // `--ativacao` está ligado.
        if cfg!(debug_assertions) {
            debug_assert!(
                states[..n].iter().flat_map(|s| s.phys.pos.iter()).all(|v| v.is_finite()),
                "NaN/Inf detectado em posições antes do submit GPU (tick {tick_idx})"
            );
        }

        // ── Poses retina ──────────────────────────────────────────────────────
        let pose_raw: Vec<PoseRaw> = (0..n).map(|i| {
            let p = &states[i].phys;
            PoseRaw {
                pos_gaze: [p.pos[0], p.pos[1], p.pos[2], 0.0],
                dirs:     [p.fwd[0], p.fwd[2], p.lat[0], p.lat[2]],
            }
        }).collect();
        self.queue.write_buffer(&self.buf_poses_retina, 0, bytemuck::cast_slice(&pose_raw));

        // ── Poses percepção ───────────────────────────────────────────────────
        let poses_percept: Vec<PosePerception> = (0..n).map(|i| {
            let p = &states[i].phys;
            PosePerception {
                pos_x: p.pos[0], pos_y: p.pos[1], pos_z: p.pos[2], phase: 0.0,
                fwd_x: p.fwd[0], fwd_z: p.fwd[2], rgt_x: p.lat[0], rgt_z: p.lat[2],
            }
        }).collect();

        // ── Distâncias laterais/frontais (CPU) → buf_temporal[16-18] ─────────
        // 'dists' fica acessível fora do bloco de distâncias para que o bloco de
        // racing_line possa usar dist_l/r/f sem recalcular (conserta ch118/119/123
        // que antes recebiam 0.5/0.8/0.7 hardcoded).
        let tick_dists: Vec<[f32; 3]> = {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            // Expandido de 15m → 40m para paridade com perception.wgsl (BUG 1 fix).
            // Conta: 40 m ÷ 28 m/s = 1.43 s de antecipação lateral.
            let lat_samp = [1.0_f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 33.0, 40.0];
            let fwd_samp = [3.0_f32, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0];
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };
            let mut dists: Vec<[f32; 3]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let (perp_x, perp_z) = (-fz, fx);
                let mut dist_l = 40.0_f32;  // default = livre (sem parede em 40m)
                for &sd in &lat_samp {
                    if voxel_at(px - perp_x*sd, pz - perp_z*sd) == 2 { dist_l = sd; break; }
                }
                let mut dist_r = 40.0_f32;
                for &sd in &lat_samp {
                    if voxel_at(px + perp_x*sd, pz + perp_z*sd) == 2 { dist_r = sd; break; }
                }
                let mut dist_f = 50.0_f32;
                for &sd in &fwd_samp {
                    let v = voxel_at(px + fx*sd, pz + fz*sd);
                    if v == 2 || v == 3 { dist_f = sd; break; }
                }
                dists.push([dist_l, dist_r, dist_f]);
            }
            dists
        };
        // Grava distâncias no BeingState para o replay writer do laço externo.
        // Conserto 4 (2026-08-12): os campos dist_wall_l/r/f existiam no ReplayFrame
        // mas chegavam zerados porque a escrita usava ..Default::default(). O diagnóstico
        // de "por que o cérebro não freia" é impossível sem saber o que ele enxergava.
        for i in 0..n {
            states[i].dist_wall_l = tick_dists[i][0];
            states[i].dist_wall_r = tick_dists[i][1];
            states[i].dist_wall_f = tick_dists[i][2];
        }
        self.percept.update_dist_temporal(&self.queue, &tick_dists[..n]);

        // ── Racing line (CPU) → buf_temporal[10-15] ───────────────────────────
        // cpu_fwd_norm / cpu_corridor / cpu_whisker_norm agora são calculados a partir
        // das distâncias reais (tick_dists) em vez de constantes 0.5/0.8/0.7 hardcoded.
        // Isso alimenta ch118 (dist frontal normalizada), ch119 (corredor lateral) e
        // ch123 (whisker = menor lateral) que antes chegavam constantes ao cérebro.
        //
        // rv[i] NÃO é ch=108+i (não é linear): rv[0..3] alimentam ch108,109,110 mas
        // rv[3..6] pulam para ch118,119,123 (perception_temporal.wgsl:271-277).
        // Travado contra o registro único `Channel`.
        const _: () = {
            assert!(Channel::LineOffsetRacingLine.index() == 108, "rv[0] = off_norm");
            assert!(Channel::LineOffsetDeriv.index()      == 109, "rv[1] = 0.0 (derivada, ainda não implementada)");
            assert!(Channel::LineLookAheadRL.index()      == 110, "rv[2] = la_norm");
            assert!(Channel::FwdDistNorm.index()          == 118, "rv[3] = cpu_fwd_norm");
            assert!(Channel::CorridorNorm.index()         == 119, "rv[4] = cpu_corridor");
            assert!(Channel::WhiskerNorm.index()          == 123, "rv[5] = cpu_whisker_norm");
        };
        {
            let hw = self.racing_line.half_width.max(1.0);
            let mut rv: Vec<[f32; 6]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let pos2d = bevy::math::Vec2::new(p.pos[0], p.pos[2]);
                let fwd2d = bevy::math::Vec2::new(p.fwd[0], p.fwd[2]);
                let off_norm = (self.racing_line.lateral_offset_from_line(pos2d, fwd2d) / hw).clamp(-1.0, 1.0);
                let la_norm  = (self.racing_line.lookahead_offset(pos2d, 5.0) / hw).clamp(-1.0, 1.0);

                // Normalização idêntica à descrita em perception.rs:700-702:
                //   fwd_norm  = dist_f / (hw * 3.0)  — quanto espaço à frente em relação à pista
                //   corridor  = (dist_l + dist_r) / (hw * 4.0) — largura efetiva do corredor
                //   whisker   = min(dist_l, dist_r) / (hw * 1.5) — lado mais apertado
                let dist_l = tick_dists[i][0];
                let dist_r = tick_dists[i][1];
                let dist_f = tick_dists[i][2];
                let cpu_fwd_norm    = (dist_f / (hw * 3.0)).clamp(0.0, 1.0);
                let cpu_corridor    = ((dist_l + dist_r) / (hw * 4.0)).clamp(0.0, 1.0);
                let cpu_whisker_norm = (dist_l.min(dist_r) / (hw * 1.5)).clamp(0.0, 1.0);

                // Derivada do offset lateral (cpu_d_off, slot [1]): diferença em relação
                // ao tick anterior guardado em temporal[8] (prev_line_offset_norm).
                // Por ora usa 0.0 — o shader temporal lê prev_line_offset_norm do próprio
                // temporal[8] e poderia calcular internamente, mas o campo de leitura seria
                // o valor do TICK ANTERIOR (correto). O valor 0.0 é conservador e correto
                // para o tick 0; a derivada real emerge a partir do tick 1 via temporal[8].
                rv.push([off_norm, 0.0, la_norm, cpu_fwd_norm, cpu_corridor, cpu_whisker_norm]);
            }
            self.percept.update_racing_line_temporal(&self.queue, &rv[..n]);
        }

        // ── Propriocepção → buf_temporal[20-31] ───────────────────────────────
        // prop[i] alimenta ch(64+i) no shader (perception_temporal.wgsl:324-349).
        // Travado contra o registro único `Channel` — ver nota em "Extra2" abaixo
        // sobre por que isto pega troca de índice mas não reordenação do push.
        const _: () = {
            assert!(Channel::VelAlign.index()      == 64, "prop[0] = align");
            assert!(Channel::SpeedFwd.index()      == 65, "prop[1] = speed");
            assert!(Channel::VelLat.index()        == 66, "prop[2] = lat_vel");
            assert!(Channel::VidaFrac.index()      == 67, "prop[3] = vida_frac");
            assert!(Channel::RpmNorm.index()       == 68, "prop[4] = rpm_r");
            assert!(Channel::Gear.index()          == 69, "prop[5] = current_gear");
            assert!(Channel::TireTemp.index()      == 70, "prop[6] = tire_temp");
            assert!(Channel::TempMotor.index()     == 71, "prop[7] = temp_motor");
            assert!(Channel::RiscoCinetico.index() == 72, "prop[8] = risco_cinetico");
            assert!(Channel::OnGrass.index()       == 73, "prop[9] = on_grass");
            assert!(Channel::SteerAbs.index()      == 74, "prop[10] = |steer|");
            assert!(Channel::Estagnacao.index()    == 75, "prop[11] = estagnado");
        };
        {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            let voxel_at_cpu = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };
            let mut prop: Vec<[f32; 12]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let vel_dot = p.vel[0]*p.fwd[0] + p.vel[2]*p.fwd[2];
                let align = if speed > 0.1 { (vel_dot / speed).clamp(-1.0, 1.0) } else { 1.0 };
                let lat_vel = p.vel[0]*p.lat[0] + p.vel[2]*p.lat[2];
                let rpm_r = if p.max_rpm > 0.0 { (p.current_rpm / p.max_rpm).clamp(0.0,1.0) } else { 0.0 };
                let tt: f32 = (0..4).map(|j| p.wheels[j*2+1][0]).sum::<f32>() / 4.0 / 120.0;
                let voxel = voxel_at_cpu(p.pos[0], p.pos[2]);
                let on_grass = if voxel == 0 { 1.0_f32 } else { 0.0 };
                // ── PROPRIOCEPÇÃO: quatro slots eram CONSTANTES ────────────────
                //
                // `prop[3]=1.0` (HP sempre cheio), `prop[7]=0`, `prop[8]=0`, `prop[11]=0`
                // viravam os canais 67, 71, 72 e 75 cravados — 4 dos 12 proprioceptivos
                // que a auditoria de visão pegou saturados em 127. Canal constante é peso
                // morto em L1 e, pior, é um estado corporal que o cérebro NÃO sente.
                //
                // Isso importa além da limpeza: medimos que neuromodulação só tem teto de
                // ganho quando existe corpo ACOPLADO — estado interno lento que faz o ótimo
                // de comportamento variar. Sem acoplamento o ganho é zero. Os estados abaixo
                // já eram calculados e simplesmente não chegavam a lugar nenhum.
                //
                //   67 ← vida restante   (era HP fixo em 1,0): o ser morre sem saber que vai
                //   71 ← temperatura do motor: acima de 105 °C a potência cai
                //        (physics.wgsl:402) — muda o ótimo de acelerar, τ ≈ 500-1000 ticks
                //   72 ← RISCO CINÉTICO: velocidade × curvatura aparente da retina.
                //        Antes: temperatura do óleo (constante em pista plana, auditada como
                //        canal morto). O ser precisa aprender a FREAR antes da curva — para isso
                //        o sinal de "velocidade alta + curva fechada à frente" tem de chegar
                //        ao cérebro em um canal dedicado. Conserto 3 (2026-08-12).
                //        edge_row6 e edge_row7 medem densidade de borda horizontal linha-a-linha:
                //        a diferença entre elas sobe quando a perspectiva da curva aparece
                //        (borda superior vs inferior divergem). Multiplicado pela velocidade
                //        normalizada: alto = velocidade + curvatura próxima = freiar agora.
                //   75 ← estagnação: quantos ticks sem avançar
                let vida_frac = (states[i].life_left as f32 / life_cap().max(1) as f32)
                    .clamp(0.0, 1.0);
                let temp_motor = ((p.engine_temp - 20.0) / 130.0).clamp(0.0, 1.0);
                // Risco cinético = velocidade × curvatura aparente.
                // curv_aparente: diferença de edge density entre linhas 6 e 7 da retina.
                // Quanto maior a diferença, mais a perspectiva da parede está "virando" —
                // sinal de curva à frente. Multiplicado por speed_norm para que seja ZERO
                // em linha reta rápida e ALTO em curva fechada + velocidade alta.
                let curv_aparente = (states[i].edge_row6 - states[i].edge_row7).abs();
                let risco_cinetico = (speed / 30.0).clamp(0.0, 1.0) * curv_aparente;
                let estagnado  = (states[i].stagnation_ticks as f32 / 64.0).clamp(0.0, 1.0);
                prop.push([
                    align, speed, lat_vel, vida_frac,
                    rpm_r, p.current_gear as f32, tt.clamp(0.0,1.0), temp_motor,
                    risco_cinetico, on_grass, p.steer.abs().clamp(0.0,1.0), estagnado,
                ]);
            }
            self.percept.update_proprioception_temporal(&self.queue, &prop[..n]);
        }

        // ── Reverse timer + CfC percepção → buf_temporal[32-48] ──────────────
        {
            // Deduplicado (E3, REFACTOR_PLAN.md §2.3) para
            // universe::config::iara::reverse_maneuver — ver doc do módulo: a constante
            // local `REVERSE_TRIGGER_DIST=3.0` daqui era MORTA (nunca lida; o gatilho
            // abaixo usava o literal `4.0` hardcoded), achado ao unificar.
            use universe::config::iara::reverse_maneuver as revcfg;

            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };

            let mut extra: Vec<[f32; 17]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (perp_x, perp_z) = (-fz, fx);
                let mut fwd_dist = 30.0_f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    let v = voxel_at(px + fx*sd, pz + fz*sd);
                    if v == 2 || v == 3 { fwd_dist = sd; break; }
                }

                // ── PILOTO SENSORIAL (pilots.rs) ────────────────────────────────
                //
                // Leonardo: *"roda uma bateria onde não tem cérebro, o carro deve sobreviver
                // por sensores, ver distância das paredes em ambos os lados, sempre tentando
                // centralizar no centro da pista e fazer as curvas de forma correta."*
                //
                // O piloto fantasma usa a racing line — um oráculo que o ser NÃO possui,
                // então ele mede o corpo, não a informação. O sensorial usa só raios de
                // parede, que é exatamente o que a retina do ser enxerga: mede o TETO DA
                // INFORMAÇÃO SENSORIAL. Se um controlador de vinte linhas completa a volta
                // com esses sensores, os dados bastam e a culpa é do cérebro.
                if PILOTO_SENSOR.load(Ordering::Relaxed) != 0.0 {
                    let g = PILOTO_SENSOR.load(Ordering::Relaxed);
                    let (steer, thr, brk) = crate::pilots::piloto_sensorial(
                        g, px, pz, fx, fz, perp_x, perp_z, speed, fwd_dist, &voxel_at,
                    );
                    states[i].sensor_steer = steer;
                    states[i].sensor_thr = thr;
                    states[i].sensor_brk = brk;
                }

                // Fan-ray para o piloto fantasma (pilots.rs): 7 direções
                // (frente + ±15°/30°/45°). Calculado aqui onde `voxel_at` está vivo;
                // consumido no bloco do piloto. A curva aparece como parede diagonal
                // ANTES de aparecer frontalmente — o fan detecta a curva antes do frontal
                // puro.
                if PILOTO_GANHO.load(Ordering::Relaxed) != 0.0 {
                    states[i].phantom_d_min = crate::pilots::fan_ray_d_min(px, pz, fx, fz, &voxel_at);
                }

                if states[i].post_reverse_lockout > 0.0 {
                    states[i].post_reverse_lockout = (states[i].post_reverse_lockout - dt).max(0.0);
                }

                // Trigger de ré: parede frontal <4m E speed baixa E estagnado E sem lockout.
                {
                    let rev = states[i].reverse_timer;
                    let stag = states[i].stagnation_ticks;
                    let lockout = states[i].post_reverse_lockout;
                    if rev <= 0.0 && lockout <= 0.0
                        && fwd_dist < revcfg::TRIGGER_DIST_M
                        && speed < revcfg::TRIGGER_SPEED_MAX_MS
                        && stag > revcfg::STAGNATION_TICKS
                    {
                        // Anti-encalhe ESCALONADO (ver doc em types.rs::re_falhas):
                        // re-encalhe em <5 s = a ré anterior não escapou → dobra a
                        // duração (até 3×) e o volante alterna de lado durante a ré.
                        if states[i].re_desde_saida < 5.0 {
                            states[i].re_falhas = states[i].re_falhas.saturating_add(1);
                        } else {
                            states[i].re_falhas = 0;
                        }
                        let escala = 1.0 + (states[i].re_falhas.min(4) as f32) * 0.75;
                        states[i].reverse_timer = revcfg::MIN_DURATION_S * escala;
                        states[i].stagnation_ticks = 0;
                    }
                    states[i].re_desde_saida += dt;
                }
                // Decremento e saída de ré.
                {
                    let rev = states[i].reverse_timer;
                    if rev > 0.0 {
                        let nr = rev - dt;
                        if fwd_dist > revcfg::EXIT_CLEAR_M || nr > revcfg::MAX_DURATION_S {
                            states[i].reverse_timer = 0.0;
                            states[i].re_desde_saida = 0.0;
                            states[i].post_reverse_lockout = revcfg::POST_LOCKOUT_S;
                        } else {
                            states[i].reverse_timer = nr.max(0.0);
                        }
                    }
                }

                // val[1..17] (cfc_hidden) fica em ZERO deliberadamente: ch192-207 agora
                // são escritos diretamente por `retina_digest.wgsl` (que roda no mesmo
                // encoder, lendo o estado do CfC direto da VRAM — sem passar pela CPU).
                // Este slot de `update_extra_temporal` continua existindo só porque a
                // função é COMPARTILHADA com gpu_loop.rs/gpu_report.rs (que ainda rodam
                // o CfC na CPU); em gpu_evolve, só o slot [0] (reverse_timer) é real.
                let mut val = [0.0f32; 17];
                val[0] = states[i].reverse_timer;
                extra.push(val);
            }
            self.percept.update_extra_temporal(&self.queue, &extra[..n]);

            // ── Extra2: slip, dopamina, dor, stress, edge (slots 49-57) ───────
            //
            // Canais reprovados pela auditoria que têm dado disponível no host:
            //   ch87  = slip angle (cross2D fwd × vel_norm) — calculado aqui da física
            //   ch124 = dor de rodopio (|av_y|>1.5 && speed<2.0) — física
            //   ch125 = estresse de proximidade O(n²) — posições dos seres
            //   ch126 = joy_tick — calculado no odômetro, tick anterior (1 tick de atraso)
            //   ch127 = joy_streak — idem
            //   ch99  = edge density linha 6 — calculado no readback anterior (1 tick atraso)
            //   ch100 = edge density linha 7 — idem
            //   ch121 = steer_mem (phys.steer) — física
            //   ch122 = brake_mem (phys.brake_cmd) — física
            //
            // Os literais acima são travados em tempo de compilação contra o registro
            // único `Channel` (src/perception/channels.rs) — a primeira versão desse
            // registro tinha ch124-127 TROCADOS (Spinning/CompStress/JoyTick/JoyStreak
            // em ordem errada) e só foi pega porque a ordem aqui embaixo (extra2.push)
            // é a fonte real. Se alguém reordenar `extra2.push([...])` sem atualizar os
            // comentários `[N] → chXXX`, esses `assert!` continuam vendo o Channel certo
            // (eles não leem a ordem do array, só travam os literais desta lista de
            // comentários contra o enum) — a defesa real contra reordenar o push está
            // no teste de runtime `--auditar-visao`, que mede variância por índice.
            const _: () = {
                assert!(Channel::SlipAngle.index()  == 87,  "ch87 = slip_angle");
                assert!(Channel::JoyTick.index()    == 126, "ch126 = joy_tick");
                assert!(Channel::JoyStreak.index()  == 127, "ch127 = joy_streak");
                assert!(Channel::Spinning.index()   == 124, "ch124 = is_spinning (dor de rodopio)");
                assert!(Channel::CompStress.index() == 125, "ch125 = comp_stress (proximidade)");
                assert!(Channel::EdgeRow6.index()   == 99,  "ch99  = edge_row6");
                assert!(Channel::EdgeRow7.index()   == 100, "ch100 = edge_row7");
                assert!(Channel::SteerMem.index()   == 121, "ch121 = steer_mem");
                assert!(Channel::BrakeMem.index()   == 122, "ch122 = brake_mem");
            };
            {
                // Posições para comp_stress O(n²): 2304 pares para n=48, irrelevante.
                let positions: Vec<[f32; 2]> = (0..n)
                    .map(|i| [states[i].phys.pos[0], states[i].phys.pos[2]])
                    .collect();

                let mut extra2: Vec<[f32; 9]> = Vec::with_capacity(n);
                for i in 0..n {
                    let p = &states[i].phys;
                    let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();

                    // slip_signal: cross product 2D entre forward e vel normalize.
                    // Mesma fórmula de motor.rs:1607-1613. Positivo = drift à direita.
                    let slip_signal = if speed > 1.0 {
                        let inv_s = 1.0 / speed;
                        let vx = p.vel[0] * inv_s;
                        let vz = p.vel[2] * inv_s;
                        // f × v: fwd_x*vel_z - fwd_z*vel_x
                        p.fwd[0] * vz - p.fwd[2] * vx
                    } else { 0.0 };

                    // is_spinning: |av_y| > 1.5 rad/s E speed < 2.0 m/s (motor.rs:1383).
                    let is_spinning = if p.av_y.abs() > 1.5 && speed < 2.0 { 1.0f32 } else { 0.0 };

                    // comp_stress O(n²): distância ao ser mais próximo em raio 20 m à frente.
                    // motor.rs:1359-1378: 0=líder (ou ninguém perto), 0.5=meio pelotão, 1.0=alguém <20m.
                    //
                    // BUG encontrado e corrigido (2026-08-14, FASE CANAIS): a versão anterior
                    // usava `if i > 0 {0.5} else {0.0}` como piso — um artefato do ÍNDICE do
                    // ser (o ser 0 sempre partia de 0.0, todos os outros de 0.5), sem relação
                    // com proximidade real. Como a auditoria de visão rastreia UM ser vivo ao
                    // longo do tempo (tipicamente o de índice mais baixo), esse ser lia
                    // comp_stress preso em {0.0, 1.0} — nunca 0.5 — e o canal parecia
                    // CONSTANTE sempre que ninguém chegava a <20m dele (o caso comum numa
                    // população espalhada), mesmo com o dado corretamente "alimentado". Agora
                    // o piso 0.5 depende de proximidade FÍSICA (alguém em até 40m, à frente ou
                    // não) — igual para todos os seres, sem depender de posição no array.
                    let (px, pz) = (positions[i][0], positions[i][1]);
                    let (fx, fz) = (p.fwd[0], p.fwd[2]);
                    let mut has_near_ahead = false;
                    let mut min_dist_any: f32 = f32::MAX;
                    for j in 0..n {
                        if j == i { continue; }
                        let dx = positions[j][0] - px;
                        let dz = positions[j][1] - pz;
                        let dist = (dx*dx + dz*dz).sqrt();
                        if dist < min_dist_any { min_dist_any = dist; }
                        if dist < 20.0 {
                            let dot = (dx * fx + dz * fz) / dist.max(0.01);
                            if dot > 0.3 { has_near_ahead = true; }
                        }
                    }
                    // 0.0=ninguém em 40m, 0.5=alguém no pelotão (<40m, qualquer direção),
                    // 1.0=alguém <20m à frente no cone de visão.
                    let comp_stress = if has_near_ahead { 1.0f32 }
                        else if min_dist_any < 40.0 { 0.5 }
                        else { 0.0 };

                    extra2.push([
                        slip_signal.clamp(-1.0, 1.0),   // [0] → ch87
                        states[i].joy_tick,              // [1] → ch126
                        states[i].joy_streak / 5.0,      // [2] → ch127 (normalizado)
                        is_spinning,                     // [3] → ch124
                        comp_stress,                     // [4] → ch125
                        states[i].edge_row6,             // [5] → ch99 (tick anterior)
                        states[i].edge_row7,             // [6] → ch100 (tick anterior)
                        p.steer.clamp(-1.0, 1.0),        // [7] → ch121
                        p.brake_cmd.clamp(0.0, 1.0),     // [8] → ch122
                    ]);
                }
                self.percept.update_extra2_temporal(&self.queue, &extra2[..n]);
            }
        }

        // ── Encoder único ─────────────────────────────────────────────────────
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evolve-tick"),
        });

        // Retina.
        let mut pitches_arr = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches_arr[row / 4][row % 4] = pitch(row);
        }
        let retina_params = RetinaParamsGpu {
            grid_size: self.grid_size,
            n_beings:  n as u32,
            cell_size: self.cell_size,
            fov_half:  (universe::visao::fov_deg() / 2.0).to_radians(),
            pitches:   pitches_arr,
        };
        let buf_rp = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("evolve-rp"),
            contents: bytemuck::bytes_of(&retina_params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let bg_retina = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.layout_retina,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_rp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_poses_retina.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_retina_out.as_entire_binding() },
            ],
        });

        self.percept.copy_retina_prev(&mut enc, &self.buf_retina_out);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("evolve-retina-pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }

        // Percepção.
        self.percept.bind_and_dispatch(&mut enc, &self.device, &self.queue,
                                       &poses_percept[..n], &self.buf_retina_out);

        // CfC de percepção + min_wall_ahead + edge_row6/7 — TUDO na GPU agora.
        //
        // Roda DEPOIS de `percept.bind_and_dispatch` (que zera `buf_output` antes de
        // escrever os canais espaciais/temporais) porque este kernel escreve ch192-207
        // no MESMO `buf_output` via atomicOr — escrever antes do clear perderia esses
        // canais. Roda ANTES do cérebro, para que `brain.bind_and_dispatch` já veja
        // ch192-207 preenchidos. Substitui o antigo readback de 340 KB/tick (retina
        // inteira) + laço Rust — ver `gpu::retina_digest` para o antes/depois completo.
        let extra_bytes = self.retina_digest.bind_and_dispatch(
            &mut enc, &self.device, &self.queue, n, &self.buf_retina_out, self.percept.buf_output(),
        );

        // Cérebro rápido — só dispara a cada `brain_skip` ticks.
        //
        // A percepção e a física continuam a 64 Hz: o que muda é a frequência de DECISÃO,
        // não a do mundo. Nos ticks pulados o ser age com a última intenção, exatamente
        // como um organismo cuja via motora é mais lenta que a sensorial.
        self.brain_tick = self.brain_tick.wrapping_add(1);
        // Dispara também quando ainda NÃO existe intenção anterior utilizável.
        //
        // Sem esta condição, com skip>1 os primeiros ticks devolviam `intents` vazio, a
        // guarda `intents.len() != n` abortava o tick e a geração inteira morria: medido
        // com skip=4 → 0 m, 100% de mortalidade, 0,0 s. Sem a guarda, teria sido pior —
        // o número sairia bonitinho e eu concluiria que "skip alto faz o modelo morrer",
        // quando o que morre é a execução.
        let brain_fires = self.brain_skip <= 1
            || self.last_intents.len() != n
            || (self.brain_tick % self.brain_skip) == 0;
        if brain_fires {
            self.brain.bind_and_dispatch(&mut enc, n, &[], dt,
                                         Some(self.percept.buf_output()));
        }

        // Cérebro lento — colhe 48 canais da onda que já está na GPU e roda.
        // Encadeado no MESMO encoder: nenhum readback intermediário, nenhuma volta à CPU.
        self.slow.dispatch_gather(&mut enc, self.percept.buf_output(), n);
        self.slow.dispatch(&mut enc, n, dt);

        // Submit único.
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Leitura do intent.
        // ── AUDITORIA DA VISÃO ────────────────────────────────────────────────
        // Acumula dados ao longo de TODA a corrida (um tick por chamada) para
        // poder medir variância temporal por canal. Reporta ao final da corrida.
        if AUDITAR.load(Ordering::Relaxed) {
            use universe::gpu::perception::WAVE_U32S;
            let bytes = (n * WAVE_U32S * 4) as u64;
            let ondas = self.percept.read_after_submit(&self.device, n, bytes);
            if ondas.len() == n {
                // Escolhe o primeiro ser VIVO para rastrear temporalmente.
                // Rastrear ser[0] fixo produz σ_temporal=0 se ele morrer cedo.
                let ser_vivo = states[..n].iter().position(|s| s.life_left > 0.0).unwrap_or(0);
                let n_vivos  = states[..n].iter().filter(|s| s.life_left > 0.0).count();
                auditoria_acumular(&ondas, n, ser_vivo, n_vivos);
            }
        }

        let mut intents = if brain_fires {
            let v = self.brain.read_intent_after_submit(n);
            if v.len() == n { self.last_intents = v.clone(); }
            v
        } else {
            // Reusa a decisão anterior. Se ainda não houve nenhuma (primeiro tick com
            // skip > 1), cai em zero — que é "não faz nada", não um comando inventado.
            self.last_intents.clone()
        };
        let slow_out = self.slow.read_outputs(n);
        // ── M4: MAESTRO DE REGIÕES (ver REGIOES_CEREBRAIS.md, spec M4) ─────────
        // gate_b = 1 + 0,5·tanh(w·slow_out + b), atenuado pela RESERVA da região;
        // usar o gate longe de 1,0 GASTA reserva (burnout), que repõe com τ=30 s.
        if NEUROMOD_REGIOES.load(Ordering::Relaxed)
            && !self.gates_w.is_empty() && slow_out.len() == n
        {
            let dt_nm = 1.0f32 / 64.0;
            let mut flat: Vec<f32> = Vec::with_capacity(n * 4);
            for i in 0..n {
                let so = slow_out[i];
                let w = &self.gates_w[i * 16..(i + 1) * 16];
                for b in 0..4 {
                    let z = w[b * 4] * so[0] + w[b * 4 + 1] * so[1] + w[b * 4 + 2] * so[2] + w[b * 4 + 3];
                    let g_raw = 1.0 + 0.5 * z.tanh();
                    let r = &mut states[i].nm_reserva[b];
                    let uso = (g_raw - 1.0).abs();
                    *r = (*r + ((1.0 - *r) / 30.0 - 0.5 * uso * *r) * dt_nm).clamp(0.0, 1.0);
                    flat.push(1.0 + (g_raw - 1.0) * *r);
                }
            }
            self.ultimo_gate = [flat[0], flat[1], flat[2], flat[3]];
            self.brain.upload_gates(&flat, n);
        }
        // Falha silenciosa é o risco aqui: se `read_outputs` devolver vazio, o consumo
        // cai em `unwrap_or([0;3])` e o lento fica inerte sem nenhum sinal de erro.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static AVISADO: AtomicBool = AtomicBool::new(false);
            if slow_out.len() != n && !AVISADO.swap(true, Ordering::Relaxed) {
                eprintln!("[lento] AVISO: read_outputs devolveu {} de {} seres — órgão inerte",
                          slow_out.len(), n);
            }
        }
        // Amostra esparsa (o lento só recomputa a cada 8 ticks; amostrar mais é redundância
        // que infla artificialmente a razão de participação).
        if !slow_out.is_empty() && self.slow_samples.len() < 20_000 {
            for o in slow_out.iter().take(8) {
                self.slow_samples.push(vec![o[0], o[1], o[2]]);
            }
        }

        // ── Vida/parede/estagnação/combustível — dados de `retina_digest.wgsl` ────
        //
        // O CfC de percepção (ch192-207), `min_wall_ahead` e `edge_row6/7` agora são
        // calculados NA GPU (retina_digest.wgsl, dispatched acima, no mesmo encoder).
        // Aqui só descem 3 floats/ser (min_wall_ahead, edge_row6, edge_row7) em vez dos
        // 1764 pixels/ser que eram lidos e reprocessados em Rust — a equação em si
        // (legacy_encoding, col-pool, zone-pool, CfC) não muda, só migra de host para
        // kernel (ver `gpu::retina_digest` e `math::cfc_perception`).
        //
        // O laço de vida abaixo (wall_ticks/dreno, estagnação, combustível, -1 tick de
        // life_left) CONTINUA na CPU — só passa a consumir `min_wall_ahead` já pronto
        // em vez de recalculá-lo dos pixels.
        {
            let extra = self.retina_digest.read_after_submit(&self.device, n, extra_bytes);
            for i in 0..n {
                let min_wall_ahead = extra[i][0];
                // Um tick "colado" é um tick com parede a menos de WALL_NEAR_M à frente.
                // O limiar é maior que o carro (≈2 m) para pegar o encosto, não o roçar.
                // Só conta para quem ainda está vivo. Um ser sem vida congela com freio,
                // muitas vezes ENCOSTADO na parede, e continuaria contando como "colado"
                // até o fim da geração — a métrica mediria o próprio mecanismo de morte
                // em vez do comportamento. Foi o que inverteu o sinal na primeira medição
                // (dreno maior aparecia com MAIS tempo colado).
                if min_wall_ahead < WALL_NEAR_M && states[i].life_left > 0.0 {
                    states[i].wall_ticks += 1;
                    // Custo em TEMPO: cada tick colado consome vida extra. Evitar a
                    // parede compra oportunidade de andar, em vez de disputar com ela.
                    states[i].life_left -= WALL_LIFE_DRAIN.load(Ordering::Relaxed) as f64;
                }
                // FÍSICA 1 — custo de RPM alto (ver doc de CUSTO_RPM em statics.rs):
                // acima de 90% do giro, cada tick queima vida proporcional ao excesso.
                // Dá à marcha um MOTIVO além da velocidade: girar menos = viver mais.
                let custo_rpm = CUSTO_RPM.load(Ordering::Relaxed);
                if custo_rpm > 0.0 && states[i].life_left > 0.0 {
                    let rpm_frac = states[i].phys.current_rpm / states[i].phys.max_rpm.max(1.0);
                    if rpm_frac > 0.9 {
                        let excesso = ((rpm_frac - 0.9) / 0.1).min(1.5);
                        states[i].life_left -= (custo_rpm * excesso) as f64;
                    }
                }
                // E-REG1 — custo de fiação (ver CUSTO_FIACAO em statics.rs): dreno
                // constante por tick proporcional ao fio "longo" do genoma do ser.
                if !self.fiacao_custo.is_empty() && states[i].life_left > 0.0 {
                    states[i].life_left -= self.fiacao_custo[i.min(self.fiacao_custo.len()-1)] as f64;
                }
                // Custo de estagnação: parar sem avançar QUEIMA VIDA (Conserto 1, 2026-08-12).
                // Sintoma medido: ser para em 272 m com life=3010 ticks e fica ESTÁTICO até
                // hp=0 (tick 4020), sem tentar sair. Com life-earn, parar é de graça porque a
                // vida acumulada ao andar é gasta só 1/tick — o ser tem 47 s de "folga". Esse
                // dreno torna a paralisia fatal e força o cérebro a continuar tentando.
                // Threshold 128 ticks = 2 s: evita penalizar quem apenas reduz speed na curva
                // (esses têm stagnation=0 pois speed≥1.5 m/s). Análogo ao wall_drain.
                if states[i].stagnation_ticks > 128 && states[i].life_left > 0.0 {
                    states[i].life_left -= STAG_LIFE_DRAIN.load(Ordering::Relaxed) as f64;
                }
                // Combustível: consumo proporcional ao acelerador ao quadrado, como nas
                // branches que davam voltas. Só morde quando ligado (`--fuel`).
                if FUEL_ON.load(Ordering::Relaxed) && states[i].fuel > 0.0 {
                    let thr = states[i].phys.throttle.abs() as f64;
                    states[i].fuel -= (0.002 + thr * thr * 0.004) / 64.0;
                    if states[i].fuel <= 0.0 { states[i].life_left = 0.0; }
                }
                if states[i].life_left > 0.0 {
                    states[i].life_left -= 1.0;
                    states[i].ticks_alive += 1;
                }
                // Edge density das linhas 6 e 7 da retina → ch99, ch100.
                // Guardado nos states para ser entregue ao temporal NO PRÓXIMO tick (1 tick
                // de atraso) — mesmo contrato de antes, só que o valor já vem pronto.
                states[i].edge_row6 = extra[i][1];
                states[i].edge_row7 = extra[i][2];
            }
        }

        if intents.len() != n { return false; }

        // ── KAN eval inline (idêntico ao gpu_loop.rs) ─────────────────────────
        // Deduplicado (E3): usa universe::kan::eval_knots — mesma fórmula que era
        // reimplementada aqui e em gpu_loop.rs como closure local.
        let kan_eval = |out_idx: usize, x: f32| -> f32 {
            universe::kan::eval_knots(kan_knots, out_idx, x)
        };

        let mut phys_states: Vec<PhysicsStateGpu> = (0..n).map(|i| {
            let mut s = states[i].phys;
            let intent = &intents[i];
            // ── Modulação da metade LENTA (System 2) ──────────────────────────
            // Mesmos três papéis de `motor.rs`: agressão soma ao acelerador, o teto
            // limita o acelerador e o viés desloca o volante. As magnitudes (0,5 / 0,8
            // / 0,20) são as de lá — não é ajuste novo, é o mesmo contrato.
            //
            // Nota medida: `throttle_cap` esteve travado em 1,0 durante toda a vida do
            // projeto porque `slow[1]` nunca era positivo (init com 100% dos pesos
            // negativos). Só agora ele pode agir.
            // Sem vida: o ser para. Não é penalidade de pontos — é o fim do tempo dele.
            if states[i].life_left <= 0.0 {
                s.throttle = 0.0;
                s.brake_cmd = 1.0;
                s.steer = 0.0;
                s.dt = dt;
                return s;
            }

            let slow = if slow_ablated() { [0.0; 3] }
                       else { slow_out.get(i).copied().unwrap_or([0.0; 3]) };
            let aggression   = slow[0] * 0.5;
            let throttle_cap = 1.0 - slow[1].max(0.0) * 0.8;
            // Autoridade do lento sobre o volante, parametrizável.
            //
            // Era ±0,20 fixo — fraco demais para o problema medido: o volante ELEITO pela
            // seleção crava em +0,63, e ±0,20 não solta. O lento é a única parte do cérebro
            // com memória temporal (o estado que sabe "já estou na curva"). Se, com
            // autoridade real, ele soltar o volante, o teto era a arquitetura sufocando a
            // memória; se não soltar mesmo com autoridade, o teto é o cérebro pequeno.
            let steer_bias   = slow[2] * SLOW_STEER_AUTH.load(Ordering::Relaxed);

            let thr_kan   = (kan_eval(0, intent[0]).max(0.0) + aggression)
                                .clamp(0.0, throttle_cap);
            let steer_kan = (kan_eval(1, intent[1]) + steer_bias).clamp(-1.0, 1.0);
            let brk_kan   = kan_eval(2, intent[2]).max(0.0);
            // Alisamento do volante: `prev*(1-α) + novo*α`.
            //
            // Com α=0,50 o comando leva ~4 ticks para chegar a 90% de um alvo novo, o que
            // é histerese suficiente para SUSTENTAR um volante cravado mesmo que o cérebro
            // já esteja pedindo para soltar. Medido: o ser vira à direita em 74% dos
            // quadros (média +0,40) e roda em círculo dentro da pista, travando o progresso
            // em 265 m. Como parâmetro dá para separar culpa do cérebro de culpa do filtro.
            let alfa = STEER_ALPHA.load(Ordering::Relaxed);
            let steer_smooth = (states[i].prev_steer * (1.0 - alfa) + steer_kan * alfa)
                .clamp(-1.0, 1.0);
            let is_reversing = states[i].reverse_timer > 0.0;
            let in_lockout   = states[i].post_reverse_lockout > 0.0;
            let speed_now = (s.vel[0].powi(2) + s.vel[2].powi(2)).sqrt();
            // Brake anulado abaixo de 2 m/s (idêntico ao motor.rs).
            let brk_final = if speed_now < 2.0 { 0.0 } else { brk_kan };

            if is_reversing {
                // Ré ativa: override completo, pedal fixo 0.75, gear=0.
                // Volante ALTERNA de lado a cada ré re-disparada (re_falhas ímpar =
                // esquerda) — sair reto devolvia o carro ao MESMO canto (medido).
                s.throttle  = universe::config::iara::reverse_maneuver::PEDAL;
                s.steer     = if states[i].re_falhas == 0 { 0.0 }
                              else if states[i].re_falhas % 2 == 1 { -0.6 } else { 0.6 };
                s.brake_cmd = 0.0;
                s.current_gear = 0;
            } else if in_lockout {
                // Pós-ré: throttle capeado em 0.8.
                s.throttle  = thr_kan.min(0.8);
                s.steer     = steer_smooth;
                s.brake_cmd = brk_final;
                if s.current_gear <= 1 { s.current_gear = 2; }
            } else {
                // Modo normal: intent neural completo. SEM muletas.
                s.throttle  = thr_kan;
                s.steer     = steer_smooth;
                s.brake_cmd = brk_final;
                // PILOTO FANTASMA (pilots.rs): sobrepõe tudo. Não é muleta de produção —
                // é instrumento de diagnóstico, ligado só por flag, para responder "o
                // corpo consegue?".
                let pg = PILOTO_GANHO.load(Ordering::Relaxed);
                if pg != 0.0 {
                    let pos2d = bevy::math::Vec2::new(s.pos[0], s.pos[2]);
                    // Pure-pursuit controller: aponta para um ponto L metros à frente na
                    // racing line e corrige pelo heading, não pelo desvio lateral. Evita o
                    // bug de `lookahead_offset` que mistura erro longitudinal com lateral
                    // quando o carro está atrás do ponto de antecipação. L adaptativo: mais
                    // distância em alta velocidade para suavizar.
                    let l_lookahead = (speed_now * 2.5).clamp(20.0, 45.0);
                    let tgt = self.racing_line.lookahead_point(pos2d, l_lookahead);
                    let thr_base = PILOTO_THR.load(Ordering::Relaxed);
                    let (steer, thr, brk) = crate::pilots::piloto_fantasma_comando(
                        pg, thr_base,
                        s.pos[0], s.pos[2], s.fwd[0], s.fwd[2],
                        tgt.x, tgt.y,  // Vec2.y = world Z
                        speed_now, states[i].phantom_d_min,
                    );
                    s.steer = steer;
                    s.throttle = thr;
                    s.brake_cmd = brk;
                }
                if s.current_gear <= 1 { s.current_gear = 2; }
                if s.current_gear > 2 && speed_now < 3.0 { s.current_gear -= 1; }
            }
            // ── D4: DANO OCULTO (ver DANO_ESCALA em statics.rs) ───────────────
            // Acumula por giro alto e parede; degrada acelerador e direção. Aplicado
            // DEPOIS dos ramos de comando (afeta o que chega à física, inclusive na
            // ré) e ANTES do câmbio (rpm de leitura não muda no mesmo tick).
            let dano_escala = DANO_ESCALA.load(Ordering::Relaxed);
            if dano_escala > 0.0 && states[i].life_left > 0.0 {
                let rpm_frac = states[i].phys.current_rpm / states[i].phys.max_rpm.max(1.0);
                let mut inc = 0.0f32;
                if rpm_frac > 0.9 { inc += 0.00015 * ((rpm_frac - 0.9) / 0.1).min(1.5); }
                if states[i].dist_wall_f.min(states[i].dist_wall_l.min(states[i].dist_wall_r)) < 3.0 {
                    inc += 0.0004;
                }
                states[i].dano = (states[i].dano + inc * dano_escala).min(1.0);
                let d = states[i].dano;
                s.throttle = s.throttle * (1.0 - 0.5 * d);
                s.steer    = s.steer    * (1.0 - 0.3 * d);
            }
            // ── CÂMBIO (paridade com motor.rs:2451-2494) ──────────────────────
            // A física deste pipeline SEMPRE teve caixa de 6 marchas (gear_ratio em
            // physics_power_*.wgsl move a força), mas nada aqui subia de marcha: só o
            // piso "≥2" e a queda <3 m/s acima. Teto físico da marcha 2:
            // v_max = (8500+250)rpm ÷ (3,8×3,7) × 0,33 m = 21,5 m/s — e o pico real
            // medido da população era 21,9 m/s, cravado no limitador. O Bevy resolve
            // com auto-shift por RPM + override neural; portado aqui atrás de
            // `--cambio auto|neural` (padrão travado até o A/B validar).
            let modo_cambio = CAMBIO_MODO.load(Ordering::Relaxed);
            if modo_cambio > 0 && !is_reversing {
                let rpm     = states[i].phys.current_rpm;
                let max_rpm = states[i].phys.max_rpm.max(1.0);
                // Embreagem colada ≈ marcha ≥2 e v > 3,5 m/s. Bevy usa wheel_rpm>1200,
                // que em marcha 2 (ratio 12) é v > 3,33 — aproximação de mesma ordem.
                let engaged = s.current_gear >= 2 && speed_now > 3.5;
                states[i].shift_cooldown = (states[i].shift_cooldown - dt).max(0.0);
                if states[i].shift_cooldown <= 0.0 {
                    let mut req: i32 = 0;
                    let teto = CAMBIO_TETO.load(Ordering::Relaxed);
                    if engaged && rpm > max_rpm * 0.85 && s.current_gear < teto {
                        req = 1;      // auto-upshift (motor.rs:2476)
                    } else if engaged && rpm < max_rpm * 0.30 && s.current_gear > 2 {
                        req = -1;     // auto-downshift (motor.rs:2480)
                    } else if modo_cambio == 2 && engaged {
                        // Override neural por BORDA (motor.rs:2484-2487): intent[3] sobe,
                        // intent[4] desce — as duas primeiras linhas cegas de L2 ganham função.
                        if intent[3] > 0.5 && states[i].shift_up_prev <= 0.5 && s.current_gear < teto {
                            req = 1;
                        } else if intent[4] > 0.5 && states[i].shift_down_prev <= 0.5 && s.current_gear > 2 {
                            req = -1;
                        }
                    }
                    if req != 0 {
                        s.current_gear = (s.current_gear as i32 + req).clamp(2, teto as i32) as u32;
                        states[i].shift_cooldown = 0.35;
                    }
                    // prev SÓ atualiza quando a troca era possível (cooldown zerado,
                    // fora da ré): borda que sobe DURANTE cooldown/ré fica "pendente"
                    // e dispara na primeira janela válida. Atualizar sempre engolia a
                    // borda (prev virava 1 antes da janela abrir) — achado da bateria
                    // de validação P1.9, dois sítios (cooldown e pós-ré).
                    states[i].shift_up_prev   = intent[3];
                    states[i].shift_down_prev = intent[4];
                }
            }
            // Telemetria de pico (prova de vida do câmbio — ver statics.rs).
            if speed_now > VEL_PICO.load(Ordering::Relaxed) {
                VEL_PICO.store(speed_now, Ordering::Relaxed);
            }
            if s.current_gear > MARCHA_PICO.load(Ordering::Relaxed) {
                MARCHA_PICO.store(s.current_gear, Ordering::Relaxed);
            }
            // ── OVERRIDE BRUTO, o último da cadeia ────────────────────────────
            // Vem depois de TODOS os ramos (normal, ré, lockout, piloto fantasma) porque a
            // pergunta que ele responde é sobre o CORPO: dado este comando exato chegando
            // à física, o carro vira? Qualquer coisa entre o valor e o `PhysicsStateGpu`
            // seria mais uma variável a suspeitar.
            // Piloto sensorial: sobrepõe o cérebro inteiro.
            if PILOTO_SENSOR.load(Ordering::Relaxed) != 0.0 {
                s.steer     = states[i].sensor_steer;
                s.throttle  = states[i].sensor_thr;
                s.brake_cmd = states[i].sensor_brk;
                if s.current_gear <= 1 { s.current_gear = 2; }
            }
            let ke = STEER_ESCALA.load(Ordering::Relaxed);
            if ke > 0.0 { s.steer = (s.steer * ke).clamp(-1.0, 1.0); }
            let sf = STEER_FIXO.load(Ordering::Relaxed);
            if sf != 0.0 { s.steer = sf.clamp(-1.0, 1.0); }
            let tf = THR_FIXO.load(Ordering::Relaxed);
            if tf != 0.0 { s.throttle = tf.clamp(0.0, 1.0); s.brake_cmd = 0.0; }
            s.dt = dt;
            s
        }).collect();

        // ── LABORATÓRIO DE OBSERVAÇÃO (`--observar`) ──────────────────────────
        // Grava a cadeia neural COMPLETA do ser 0 neste tick, em f32 cru:
        //   [tick, speed, rpm, gear, engine_temp, throttle, steer, brake, life, joy,
        //    onda_cfc(96) canais 0..96 pós-KAN+CfC (i8 como f32),
        //    onda_kan(96) canais 96..192 pós-KAN,
        //    l1_raw(96)   acumulador L1 pré-tanh,
        //    intent(96)   final_intent pós-tanh (0..48=L2, 48..96=bypass)] = 394 f32.
        // Só quando o cérebro disparou (senão a linha repetiria a decisão anterior).
        // PILOTO POR FÓRMULA: substitui a decisão do ser 0 mantendo percepção+KAN+CfC.
        // Mesmo canal de leitura do --observar (download_debug) → X idêntico ao do solve.
        if self.piloto_formula.is_some() && brain_fires && !intents.is_empty() {
            let (_l1_raw, cfc_in, kan_in) = self.brain.download_debug();
            let mut x = [0.0f32; 192];
            x[..96].copy_from_slice(&cfc_in[..96]);
            let n_kan = kan_in.len().min(96);
            x[96..96 + n_kan].copy_from_slice(&kan_in[..n_kan]);
            if let Some(pl) = self.piloto_formula.as_mut() {
                let y = pl.agir(&x);
                let mut v = intents[0];
                v[0] = y[0]; v[1] = y[1]; v[2] = y[2];
                intents[0] = v;
                self.last_intents[0] = v;
            }
        }
        if self.obs_writer.is_some() && brain_fires {
            let (l1_raw, cfc_in, kan_in) = self.brain.download_debug();
            let p = &phys_states[0];
            let st = &states[0];
            let mut row: Vec<f32> = Vec::with_capacity(394);
            row.extend_from_slice(&[
                tick_idx as f32,
                (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt(),
                p.current_rpm,
                p.current_gear as f32,
                p.engine_temp,
                p.throttle, p.steer, p.brake_cmd,
                st.life_left as f32,
                st.joy_tick,
            ]);
            row.extend_from_slice(&cfc_in);                 // 96
            row.extend_from_slice(&kan_in[..96.min(kan_in.len())]); // 96
            row.extend_from_slice(&l1_raw);                 // 96
            row.extend_from_slice(&intents[0]);             // 96
            debug_assert_eq!(row.len(), 394);
            use std::io::Write as _;
            if let Some(w) = self.obs_writer.as_mut() {
                let _ = w.write_all(bytemuck::cast_slice(&row));
            }
        }

        // ── AMOSTRAGEM DE MÉTRICAS DE ATIVAÇÃO (E4, `--ativacao`) ───────────────
        //
        // Só roda quando `ATIVACAO_ON` está ligado E o cérebro disparou neste tick
        // (senão `intents` é uma cópia da última decisão — amostrar de novo seria
        // amostra duplicada, não amostra nova) E o tick bate no período configurado
        // (`ATIVACAO_N`, padrão 64). Custo real quando ligado:
        //   - L2 (`intents[0..48]`): ZERO round-trip extra — `intents` já foi lido de
        //     volta da GPU alguns parágrafos acima, todo tick em que o cérebro dispara,
        //     independente desta flag. Só reaproveita o `Vec` que já existe em CPU.
        //   - L1 (`buf_cfc`): round-trip extra de verdade (`download_cfc_state` monta
        //     encoder+staging+map própria) — é o único custo real desta seção, e só
        //     ocorre 1 vez a cada `ATIVACAO_N` ticks.
        // Custo quando DESLIGADO (padrão): zero — nem o `if` de dentro roda.
        if ATIVACAO_ON.load(Ordering::Relaxed) && brain_fires {
            let stride = ATIVACAO_N.load(Ordering::Relaxed).max(1);
            if (tick_idx as u32) % stride == 0 {
                // Distância à parede mais próxima (min dos 3 raios já calculados no topo
                // do tick, gravados em `states[i].dist_wall_*`) — mesmo dado que alimenta
                // ch118/119/123, reaproveitado aqui sem raio novo.
                let dist_parede = |i: usize| states[i].dist_wall_l
                    .min(states[i].dist_wall_r)
                    .min(states[i].dist_wall_f);
                if intents.len() == n {
                    for (i, row) in intents.iter().enumerate() {
                        self.l2_samples.push(row[0..48].to_vec());
                        // L2 nunca pula por NaN (só L1 tem essa guarda) — lockstep
                        // garantido: um push de contexto por push de amostra.
                        self.regiao_l2.push(contexto_regiao(&self.track_curvatura, &states[i].phys, dist_parede(i)));
                    }
                }
                // Estado CfC de L1 — único readback extra desta seção.
                let cfc = self.brain.download_cfc_state(n);
                for (i, row) in cfc.iter().enumerate() {
                    if row.iter().any(|v| !v.is_finite()) {
                        eprintln!(
                            "[ativacao] ERRO: NaN/Inf no estado L1 (CfC) do ser {i} no tick {tick_idx} — amostra descartada."
                        );
                        self.nan_seres.insert(i);
                        continue;
                    }
                    self.l1_samples.push(row.clone());
                    // Mesmo `continue` acima pula ESTE push também — l1_samples e
                    // regiao_l1 crescem sempre juntos, nunca desalinham.
                    self.regiao_l1.push(contexto_regiao(&self.track_curvatura, &states[i].phys, dist_parede(i)));
                }
                // Comandos motores do "campeão" (ser 0) — para entropia de política.
                // `phys_states[0]` já tem o comando FINAL (pós axiom_policy/piloto/
                // override), antes da física consumi-lo — é o que realmente foi
                // mandado ao corpo neste tick.
                if n > 0 {
                    let p = &phys_states[0];
                    self.policy_samples.push([p.throttle, p.steer, p.brake_cmd]);
                }
            }
        }

        // Atualiza steer e stagnation.
        for i in 0..n {
            if states[i].reverse_timer > 0.0 {
                states[i].prev_steer = 0.0;
            } else {
                states[i].prev_steer = phys_states[i].steer;
            }
            let speed = (phys_states[i].vel[0].powi(2) + phys_states[i].vel[2].powi(2)).sqrt();
            if speed < 1.5 { states[i].stagnation_ticks += 1; } else { states[i].stagnation_ticks = 0; }
        }

        let ok = self.phys_gpu.step(&mut phys_states);
        if !ok { return false; }
        for i in 0..n {
            states[i].phys = phys_states[i];
            // ── Rotação de yaw: aplica av_y * dt à orientação do ser ─────────
            // O kernel de física escreve `out.fwd = fwd` (sem rotacionar — esse
            // passo ficava no ECS Bevy). No modo headless fazemos manualmente:
            //   rotate_y(θ): x' = x·cos(θ) - z·sin(θ),  z' = x·sin(θ) + z·cos(θ)
            // Convenção: av_y > 0 = rotação anti-horária vista de cima (+Y) —
            // igual à CPU (physics.rs:760: transform.rotate_y(body.angular_velocity.y * dt)).
            let theta = states[i].phys.av_y * dt;
            let (sin_t, cos_t) = theta.sin_cos();
            let fwd = states[i].phys.fwd;
            let lat = states[i].phys.lat;
            states[i].phys.fwd[0] = fwd[0] * cos_t - fwd[2] * sin_t;
            states[i].phys.fwd[2] = fwd[0] * sin_t + fwd[2] * cos_t;
            states[i].phys.lat[0] = lat[0] * cos_t - lat[2] * sin_t;
            states[i].phys.lat[2] = lat[0] * sin_t + lat[2] * cos_t;
            // Normaliza para evitar deriva numérica acumulada.
            let fn_ = (states[i].phys.fwd[0].powi(2) + states[i].phys.fwd[2].powi(2)).sqrt().max(1e-6);
            let ln_ = (states[i].phys.lat[0].powi(2) + states[i].phys.lat[2].powi(2)).sqrt().max(1e-6);
            states[i].phys.fwd[0] /= fn_; states[i].phys.fwd[2] /= fn_;
            states[i].phys.lat[0] /= ln_; states[i].phys.lat[2] /= ln_;
        }

        // memory0 → temporal.
        let mem0: Vec<f32> = (0..n).map(|i| intents[i][0].clamp(0.0, 1.0)).collect();
        self.percept.update_memory0(&self.queue, &mem0[..n]);

        true
    }

    /// Reseta CfC de percepção (estado recorrente) para todos os seres.
    /// Chamado entre gerações para que os filhos comecem sem memória.
    pub fn reset_cfc(&mut self, n: usize) {
        self.retina_digest.reset(&self.queue, n);
    }

    /// Troca a pista sem recriar device/pipelines/bind-groups — só reescreve os
    /// buffers estáticos (grade/fluxo/alturas/heightmap) via `write_buffer` (P1:
    /// rotação de pistas por geração + avaliação held-out, ver
    /// docs/PLANO_EVOLUCAO.md §P1).
    ///
    /// NÃO chamado em nenhum caminho quando `--pistas`/`--pista-held-out` estão
    /// ausentes — o binário sem essas flags nunca invoca este método, então o
    /// comportamento (e o A/B em andamento) fica bit-a-bit idêntico ao de antes
    /// desta mudança.
    ///
    /// Só é seguro porque `VoxelWorld::new` usa `size=320`/`cell_size=3.0`
    /// CONSTANTES independentes da seed (world.rs) — nenhum buffer estático muda de
    /// tamanho entre pistas, então dá pra sobrescrever em vez de realocar. Custo:
    /// ~400 KB de upload total (grid×3 + heights + hmap), sub-milissegundo — ver a
    /// estimativa em docs/PLANO_EVOLUCAO.md §P1 (irrelevante perto dos ~19,2 s/gen
    /// medidos no P0).
    pub fn reload_world(&mut self, world: &VoxelWorld) {
        debug_assert_eq!(world.size as u32, self.grid_size,
            "reload_world: pista nova tem grade de tamanho diferente — buffer não realoca");
        debug_assert!((world.cell_size - self.cell_size).abs() < 1e-6,
            "reload_world: pista nova tem cell_size diferente — buffer não realoca");

        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        self.queue.write_buffer(&self.buf_grid, 0, bytemuck::cast_slice(&grid_u32));

        self.percept.reload_world(&self.queue, world);

        let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
        self.phys_gpu.reload_world(&grid_u8, &world.height_map);

        // Cópias CPU que ficariam presas na pista antiga se esquecidas aqui — ver
        // risco correspondente em docs/PLANO_EVOLUCAO.md §P1: `racing_line`
        // alimenta os canais sensoriais 108-110 (linha ideal), `grid_cpu` alimenta
        // o voxel-sob-o-ser (dopamina/joy_tick, colisão CPU-side dos raios de
        // parede em `tick()`).
        self.racing_line = world.racing_line.clone();
        self.grid_cpu = grid_u32;

        // κ(x,z) da pista NOVA — sem isto, depois de uma troca de pista (`--pistas`/
        // `--pista-held-out`), `--regioes` continuaria medindo curvatura da pista
        // ANTERIOR contra amostras de ativação da pista atual: números com aparência
        // válida, silenciosamente errados.
        let wps: Vec<(f32, f32)> = world.waypoints_world.iter().map(|p| (p.x, p.y)).collect();
        self.track_curvatura = CurvaturaPista::medir(&wps);
    }
}
