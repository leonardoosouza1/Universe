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

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use universe::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS};
use universe::config::iara::vision::{RETINA_ROWS, RETINA_TOTAL, RETINA_COLS};
use universe::engine::SovereignEngine;
use universe::gpu::{Pose, RetinaGpu};
use universe::gpu::brain::{BrainGpu, BrainInput};
use universe::gpu::slow_brain::SlowBrainGpu;
use std::sync::atomic::Ordering;

/// Ablação da metade lenta: quando ligada, as três saídas do lento são zeradas antes
/// do consumo. O órgão continua rodando (mesmo custo, mesmo estado, mesma evolução dos
/// pesos) — só não INFORMA. É a ablação pareada correta: isola o efeito do sinal, não
/// o efeito de existir. Comparar contra um binário sem o órgão mudaria duas coisas.
static SLOW_ABLATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn slow_ablated() -> bool { SLOW_ABLATED.load(std::sync::atomic::Ordering::Relaxed) }
use universe::gpu::evolution::{Genome, EvoParams, GenResult, evolve, gen_stats, linear_slope};
use universe::gpu::perception::{PerceptionGpu, PosePerception, TEMPORAL_STRIDE};
use universe::gpu::physics::{
    PhysicsGpu, PhysicsStateGpu, pack_config_flags, WheelGpu,
};
use universe::world::VoxelWorld;
use universe::simulation::racing_line::RacingLine;
use universe::visao::pitch;
use universe::perception::RetinaPerception;
use universe::kan::KAN_OUT_KNOTS;
use universe::evolution::dna_genesis::default_kan_knots;

// ── Tipos locais (espelhando gpu_loop.rs) ────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RetinaParamsGpu {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    fov_half:  f32,
    pitches:   [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PoseRaw {
    pos_gaze: [f32; 4],
    dirs:     [f32; 4],
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ── Estado por ser (CPU-side) ─────────────────────────────────────────────────

struct BeingState {
    phys:               PhysicsStateGpu,
    prev_steer:         f32,
    stagnation_ticks:   u32,
    /// Ticks com parede dentro de `WALL_NEAR_M` à frente. Vira custo de fitness quando
    /// `--wall-cost` > 0; com 0 é só telemetria, e serve de baseline pareado.
    wall_ticks:         u32,
    /// Ticks em que o ser esteve VIVO. Sem isto a velocidade média fica errada: dividir a
    /// distância pelo tempo total da geração subestima quem morreu cedo, que é exatamente
    /// quem o experimento de vida-ganha precisa medir.
    ticks_alive:        u32,
    /// Ticks de vida restantes. Encostar na parede CONSOME vida, não pontos.
    ///
    /// Lei da Moeda Cruzada (validada no ecossistema): toda ação protetora tem de custar
    /// em moeda DIFERENTE do que protege, senão a política degenerada domina. Medido aqui:
    /// cobrando a parede em DISTÂNCIA (mesma moeda da fitness), o ser aprendeu a não andar
    /// — tempo colado caiu de 36% para 7,7%, mas a mediana caiu de 126,7 m para 46,0 m e a
    /// mortalidade triplicou. Cobrando em TEMPO, evitar a parede COMPRA oportunidade de
    /// andar em vez de disputar com ela.
    /// Vida restante em ticks, como f64.
    ///
    /// Era i64 e o ganho `(passo * earn) as i64` TRUNCAVA: a 10,14 m/s o passo é
    /// 0,158 m/tick e ×6 dá 0,95 → vira 0. Com earn=e o ser só ganhava vida andando mais
    /// de 1/e metros por tick (acima de 10,7 m/s para e=6); abaixo disso ganhava NADA.
    life_left:          f64,
    reverse_timer:      f32,
    post_reverse_lockout: f32,
}

fn pos_of(p: &PhysicsStateGpu) -> [f32; 3] { p.pos }

/// Constrói PhysicsStateGpu inicial para um ser na posição `start`.
/// Idêntico ao gpu_loop.rs para manter comparabilidade.
fn make_physics_state(start: [f32; 3], fwd: [f32; 3]) -> PhysicsStateGpu {
    let normal_load = 1350.0 * 9.81 / 4.0;
    let wheel = WheelGpu::from_cpu(0.0, 0.0, 0.0, normal_load, 60.0, 0.0, 0.0);
    let lat = [fwd[2], 0.0, -fwd[0]];
    // v0 = 0 m/s: sem velocidade injetada. Movimento = decisão do cérebro.
    let v0 = 0.0_f32;
    PhysicsStateGpu {
        vel: [fwd[0] * v0, 0.0, fwd[2] * v0],
        av_y: 0.0,
        wheels: [wheel.part_a, wheel.part_b,
                 wheel.part_a, wheel.part_b,
                 wheel.part_a, wheel.part_b,
                 wheel.part_a, wheel.part_b],
        pos: [start[0], start[1].max(0.5), start[2]],
        _pad_pos: 0.0,
        fwd,
        _pad_fwd: 0.0,
        lat,
        _pad_lat: 0.0,
        max_torque: 620.0,
        current_rpm: 2000.0,
        max_rpm: 8500.0,
        throttle: 0.0,
        brake_cmd: 0.0,
        steer: 0.0,
        drive_torque_out: 0.0,
        current_gear: 3,   // 2ª marcha (convenção: 0=ré, 1=neutro, 2=1ª ...)
        shift_timer: 0.0,
        is_stalled: 0,
        starter_timer: 0.0,
        flywheel_inertia: 0.15,
        engine_temp: 90.0,
        oil_temp: 85.0,
        mass: 1350.0,
        drag_coeff: 0.32,
        wheel_base: 2.65,
        cg_height: 0.45,
        track_width: 1.55,
        lsd_lock_ratio: 0.3,
        brake_torque_max: 4000.0,
        drivetrain: 1,
        mu_peak: 1.6,
        cornering_stiffness: 55000.0,
        wheel_radius: 0.33,
        wheel_inertia: 1.5,
        optimal_temp: 0.5,
        susp_stiffness: 45000.0,
        susp_damping: 3000.0,
        susp_rest: 0.25,
        susp_max_travel: 0.12,
        _pad_before_wind: 0.0,
        wind: [0.0; 3],
        _pad_wind: 0.0,
        terrain_grip: 1.0,
        env_grip: 1.0,
        aquaplaning: 1.0,
        config_flags: pack_config_flags(true, true, true, true, true, true),
        dt: 1.0 / 64.0,
        _pad_dt: 0.0,
    }
}

// ── Contexto GPU (espelho de GpuLoop em gpu_loop.rs) ─────────────────────────
//
// Copiamos a estrutura do gpu_loop.rs quase integralmente para não alterar
// o arquivo validado. A única diferença está no `tick()`: sem muletas motoras
// (escape_ticks, curvature_governor). O cérebro comanda tudo.

struct GpuCtx {
    device:  Arc<wgpu::Device>,
    queue:   Arc<wgpu::Queue>,
    pipeline_retina: wgpu::ComputePipeline,
    layout_retina:   wgpu::BindGroupLayout,
    buf_grid:        wgpu::Buffer,
    buf_poses_retina: wgpu::Buffer,
    buf_retina_out:   wgpu::Buffer,
    grid_size:   u32,
    cell_size:   f32,
    percept:     PerceptionGpu,
    brain:       BrainGpu,
    /// Metade LENTA do cérebro (System 2): CfC-64 → Dense → Dense, 1 tick a cada 8.
    /// Estava ausente de todo o caminho GPU — `grep slow src/gpu/` voltava vazio — o que
    /// significa que "portar o cérebro para a GPU" tinha na verdade portado só a metade
    /// rápida. As três saídas modulam agressão, teto de acelerador e viés de volante.
    slow:        SlowBrainGpu,
    /// Amostras da saída do lento ao longo da geração, para MEDIR se ele merece existir.
    /// Uma linha por amostra, três colunas — alimenta `brain_capacity::participation_ratio`.
    slow_samples: Vec<Vec<f32>>,
    /// Última intenção neural, reusada nos ticks em que o cérebro não dispara.
    last_intents: Vec<[f32; 96]>,
    /// Ticks de física por decisão neural. 1 = decide todo tick (64 Hz).
    ///
    /// É a "velocidade de simulação" pelo lado do cérebro. O `motor.rs` tinha esse
    /// mecanismo adaptativo pelo cortisol e ele foi DESLIGADO com a justificativa
    /// registrada em motor.rs:1717 — "skip_n=2 em gen>=10 reduzia brain freq pra 30Hz e
    /// degradava DNA reload". Ou seja: já se sabia que existe um ótimo interior, e a
    /// resposta foi travar em 1 em vez de medir onde ele fica. Aqui vira parâmetro.
    brain_skip: u32,
    brain_tick: u32,
    racing_line: RacingLine,
    grid_cpu:    Vec<u32>,
    grid_size_cpu: u32,
    cell_size_cpu: f32,
    phys_gpu:    PhysicsGpu,
    percept_cfc: Vec<RetinaPerception>,
    buf_retina_readback: wgpu::Buffer,
    capacity:    usize,
}

impl GpuCtx {
    fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
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
        // Cérebro usa 9 storage buffers (bindings 1-9, incluindo buf_fatigue adicionado).
        limits.max_storage_buffers_per_shader_stage = 9;
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

        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("evolve-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage:    wgpu::BufferUsages::STORAGE,
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../gpu/retina.wgsl"))),
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

        let retina_readback_bytes = (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let buf_retina_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("evolve-retina-readback"),
            size:               retina_readback_bytes,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let percept_cfc: Vec<RetinaPerception> = (0..cap).map(|_| RetinaPerception::new()).collect();

        eprintln!("[gpu_evolve] Adapter: {} · cap={}", name, cap);

        Some(Self {
            device, queue,
            pipeline_retina, layout_retina,
            buf_grid, buf_poses_retina, buf_retina_out,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            percept, brain, slow,
            slow_samples: Vec::new(),
            last_intents: Vec::new(),
            brain_skip: 1,
            brain_tick: 0,
            racing_line: world.racing_line.clone(),
            grid_cpu: grid_u32,
            grid_size_cpu: world.size as u32,
            cell_size_cpu: world.cell_size,
            phys_gpu,
            percept_cfc,
            buf_retina_readback,
            capacity: cap,
        })
    }

    /// Carrega pesos dos engines na GPU.
    fn upload_engines(&self, engines: &[SovereignEngine]) {
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
    fn tick(
        &mut self,
        states:  &mut Vec<BeingState>,
        n:       usize,
        dt:      f32,
        tick_idx: usize,
        kan_knots: &[[f32; KAN_OUT_KNOTS]; 3],
    ) -> bool {
        if n == 0 || n > self.capacity { return false; }

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
        {
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
            self.percept.update_dist_temporal(&self.queue, &dists[..n]);
        }

        // ── Racing line (CPU) → buf_temporal[10-15] ───────────────────────────
        {
            let hw = self.racing_line.half_width.max(1.0);
            let mut rv: Vec<[f32; 6]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let pos2d = bevy::math::Vec2::new(p.pos[0], p.pos[2]);
                let fwd2d = bevy::math::Vec2::new(p.fwd[0], p.fwd[2]);
                let off_norm = (self.racing_line.lateral_offset_from_line(pos2d, fwd2d) / hw).clamp(-1.0, 1.0);
                let la_norm  = (self.racing_line.lookahead_offset(pos2d, 5.0) / hw).clamp(-1.0, 1.0);
                rv.push([off_norm, 0.0, la_norm, 0.5, 0.8, 0.7]);
            }
            self.percept.update_racing_line_temporal(&self.queue, &rv[..n]);
        }

        // ── Propriocepção → buf_temporal[20-31] ───────────────────────────────
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
                prop.push([
                    align, speed, lat_vel, 1.0,
                    rpm_r, p.current_gear as f32, tt.clamp(0.0,1.0), 0.0,
                    0.0, on_grass, p.steer.abs().clamp(0.0,1.0), 0.0,
                ]);
            }
            self.percept.update_proprioception_temporal(&self.queue, &prop[..n]);
        }

        // ── Reverse timer + CfC percepção → buf_temporal[32-48] ──────────────
        {
            const REVERSE_TRIGGER_DIST: f32 = 3.0;
            const REVERSE_MIN_DURATION: f32 = 0.4;
            const REVERSE_MAX_DURATION: f32 = 1.5;
            const REVERSE_EXIT_CLEAR: f32   = 2.5;
            const POST_REVERSE_LOCKOUT: f32 = 0.6;
            const STAGNATION_FOR_REVERSE: u32 = 60;
            const REV_SPEED_MAX: f32        = 1.5;

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

                if states[i].post_reverse_lockout > 0.0 {
                    states[i].post_reverse_lockout = (states[i].post_reverse_lockout - dt).max(0.0);
                }

                // Trigger de ré: parede frontal <4m E speed baixa E estagnado E sem lockout.
                {
                    let rev = states[i].reverse_timer;
                    let stag = states[i].stagnation_ticks;
                    let lockout = states[i].post_reverse_lockout;
                    if rev <= 0.0 && lockout <= 0.0
                        && fwd_dist < 4.0
                        && speed < REV_SPEED_MAX
                        && stag > STAGNATION_FOR_REVERSE
                    {
                        states[i].reverse_timer = REVERSE_MIN_DURATION;
                        states[i].stagnation_ticks = 0;
                    }
                }
                // Decremento e saída de ré.
                {
                    let rev = states[i].reverse_timer;
                    if rev > 0.0 {
                        let nr = rev - dt;
                        if fwd_dist > REVERSE_EXIT_CLEAR || nr > REVERSE_MAX_DURATION {
                            states[i].reverse_timer = 0.0;
                            states[i].post_reverse_lockout = POST_REVERSE_LOCKOUT;
                        } else {
                            states[i].reverse_timer = nr.max(0.0);
                        }
                    }
                }

                let hidden = &self.percept_cfc[i].pipeline.state.h;
                let mut val = [0.0f32; 17];
                val[0] = states[i].reverse_timer;
                for j in 0..16 { val[1+j] = if j < hidden.len() { hidden[j] } else { 0.0 }; }
                extra.push(val);
            }
            self.percept.update_extra_temporal(&self.queue, &extra[..n]);
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

        // Cópia retina para readback do CfC.
        let retina_copy_bytes = (n * RETINA_TOTAL * 4) as u64;
        enc.copy_buffer_to_buffer(&self.buf_retina_out, 0, &self.buf_retina_readback, 0, retina_copy_bytes);

        // Submit único.
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Leitura do intent.
        let intents = if brain_fires {
            let v = self.brain.read_intent_after_submit(n);
            if v.len() == n { self.last_intents = v.clone(); }
            v
        } else {
            // Reusa a decisão anterior. Se ainda não houve nenhuma (primeiro tick com
            // skip > 1), cai em zero — que é "não faz nada", não um comando inventado.
            self.last_intents.clone()
        };
        let slow_out = self.slow.read_outputs(n);
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

        // Atualiza CfC percepção.
        {
            let slice = self.buf_retina_readback.slice(0..retina_copy_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();
            let data = slice.get_mapped_range();
            let retina_u32: &[u32] = bytemuck::cast_slice(&data);
            for i in 0..n {
                let base = i * RETINA_TOTAL;
                let mut retina_i8 = vec![0i8; RETINA_TOTAL];
                let mut min_wall_ahead = f32::MAX;
                for j in 0..RETINA_TOTAL {
                    let packed = retina_u32[base + j];
                    // Codificação ÚNICA, de `gpu::legacy_encoding`. Aqui havia uma quarta
                    // cópia da fórmula, e ela tinha DIVERGIDO: parede em rampa LINEAR
                    // (`127 - 72·d/30`) onde os outros três caminhos — world.rs,
                    // gpu/mod.rs e perception.wgsl — usam raiz (`127 - 72·√(d/30)`).
                    // A 15 m isso dava 91 contra 76: o CfC de percepção deste binário
                    // recebia parede mais LONGE do que ela está, justamente no curto
                    // alcance que decide se o ser bate.
                    let dist_m   = (packed >> 8) as f32 / 1000.0;
                    let material = (packed & 0xFF) as u8;

                    // Parede à frente: colunas centrais da retina (o terço do meio dos
                    // 120° de campo). O dado já está aqui — a retina é lida de volta a
                    // cada tick para o CfC de percepção — então medir isto é de graça.
                    if material == VOXEL_WALL {
                        let col = j % RETINA_COLS;
                        if col >= RETINA_COLS / 3 && col < 2 * RETINA_COLS / 3 {
                            min_wall_ahead = min_wall_ahead.min(dist_m);
                        }
                    }

                    retina_i8[j] = universe::gpu::legacy_encoding(universe::gpu::Pixel {
                        dist_m,
                        material,
                    });
                }
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
                if states[i].life_left > 0.0 {
                    states[i].life_left -= 1.0;
                    states[i].ticks_alive += 1;
                }
                self.percept_cfc[i].perceive(&retina_i8);
            }
            drop(data);
            self.buf_retina_readback.unmap();
        }

        if intents.len() != n { return false; }

        // ── KAN eval inline (idêntico ao gpu_loop.rs) ─────────────────────────
        let kan_eval = |out_idx: usize, x: f32| -> f32 {
            let xc = x.clamp(-2.0, 2.0);
            let u = (xc + 2.0) / 4.0;
            let fpos = u * 11.0;
            let i0 = fpos.floor() as usize;
            let i1 = (i0 + 1).min(11);
            let frac = fpos - i0 as f32;
            let a = kan_knots[out_idx][i0];
            let b = kan_knots[out_idx][i1];
            (a + (b - a) * frac).clamp(-1.0, 1.0)
        };

        const REVERSE_PEDAL: f32 = 0.75;

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
            let steer_bias   = slow[2] * 0.20;

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
                s.throttle  = REVERSE_PEDAL;
                s.steer     = 0.0;
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
                if s.current_gear <= 1 { s.current_gear = 2; }
                if s.current_gear > 2 && speed_now < 3.0 { s.current_gear -= 1; }
            }
            s.dt = dt;
            s
        }).collect();

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
        for i in 0..n { states[i].phys = phys_states[i]; }

        // memory0 → temporal.
        let mem0: Vec<f32> = (0..n).map(|i| intents[i][0].clamp(0.0, 1.0)).collect();
        self.percept.update_memory0(&self.queue, &mem0[..n]);

        true
    }

    /// Reseta CfC de percepção (estado recorrente) para todos os seres.
    /// Chamado entre gerações para que os filhos comecem sem memória.
    fn reset_cfc(&mut self, n: usize) {
        for i in 0..n.min(self.capacity) {
            self.percept_cfc[i] = RetinaPerception::new();
        }
    }
}

// ── Persistência de genomas ───────────────────────────────────────────────────

/// Salva o melhor genoma em bytes (L1 f32 LE + L2 f32 LE + KAN f32 LE).
fn save_genome(genome: &Genome, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for w in &genome.l1 { f.write_all(&w.to_le_bytes())?; }
    for w in &genome.l2 { f.write_all(&w.to_le_bytes())?; }
    for row in &genome.kan_knots {
        for k in row { f.write_all(&k.to_le_bytes())?; }
    }
    Ok(())
}

/// Tenta carregar genoma de um arquivo salvo por `save_genome`.
fn load_genome(path: &str) -> Option<Genome> {
    use std::io::Read as IoRead;
    let bytes = std::fs::read(path).ok()?;
    let n_l1 = universe::gpu::evolution::N_L1_WEIGHTS;
    let n_l2 = universe::gpu::evolution::N_L2_WEIGHTS;
    let n_kan = 3 * KAN_OUT_KNOTS;
    let expected = (n_l1 + n_l2 + n_kan) * 4;
    if bytes.len() < expected { return None; }

    let mut off = 0;
    let mut read_f32_slice = |n: usize| -> Vec<f32> {
        let end = off + n * 4;
        let v: Vec<f32> = bytes[off..end].chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        off = end;
        v
    };

    let l1 = read_f32_slice(n_l1);
    let l2 = read_f32_slice(n_l2);
    let kan_flat = read_f32_slice(n_kan);
    let mut kan_knots = default_kan_knots();
    for o in 0..3 {
        for k in 0..KAN_OUT_KNOTS {
            kan_knots[o][k] = kan_flat[o * KAN_OUT_KNOTS + k];
        }
    }
    // Arquivos de genoma anteriores à metade lenta não têm esses pesos; cair na
    // inicialização padrão preserva a compatibilidade sem fingir que o dado existia.
    let slow = universe::engine::SlowBrain::new().all_weights();
    Some(Genome { l1, l2, kan_knots, slow })
}

// ── Ponto de entrada ──────────────────────────────────────────────────────────

/// Semente do genoma inicial do ser `i` sob a semente de população `pop_seed`.
///
/// Existia em TRÊS ramos distintos (checkpoint falhou / sem checkpoint / do zero) e eu
/// semeei apenas um, sem conferir qual roda. Resultado: uma bateria inteira de 6 corridas
/// com "três sementes" que eram a MESMA população — `gen 0` idêntico até o tempo de parede
/// nas três, e um efeito pareado de 25σ que era uma amostra repetida três vezes.
///
/// Com a função única, semear é impossível de esquecer num ramo.
fn population_seed_for(pop_seed: u64, index: usize) -> u64 {
    // Multiplicador primo grande separa as famílias de sementes: populações de `pop_seed`
    // diferentes não compartilham nenhum genoma inicial.
    pop_seed.wrapping_mul(1_000_003).wrapping_add(index as u64)
}

/// Material de parede na grade de voxels (mesma ordem de `VoxelType`).
const VOXEL_WALL: u8 = 2;
/// Distância à frente abaixo da qual o ser conta como "colado" na parede.
const WALL_NEAR_M: f32 = 3.0;
/// Setores do circuito que precisam ser cruzados EM ORDEM.
///
/// 24 setores em 2.928 m dá ~122 m cada: perto o bastante para o próximo estar sempre ao
/// alcance de quem está avançando, longe o bastante para não premiar tremer no lugar.
const N_CHECKPOINTS: u32 = 24;
/// Peso do comando NOVO no alisamento do volante (1,0 = sem alisamento).
static STEER_ALPHA: AtomicF32 = AtomicF32::new();
/// Ticks de vida consumidos a mais por tick colado. Global porque o custo é aplicado
/// dentro do laço de leitura da retina, longe do parsing de argumentos.
static WALL_LIFE_DRAIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
/// Fôlego inicial quando a vida é ganha: 640 ticks = 10 s @ 64 Hz.
const LIFE_START: i64 = 640;
/// Teto de fôlego acumulável, em ticks. Padrão 1920 = 30 s.
///
/// Era constante e virou o LIMITE REAL do experimento sem eu perceber: com earn=6 os seres
/// convergiam para 9,2 m/s e viviam exatamente 1920 ticks — o teto, não a competência deles.
/// A aritmética do equilíbrio diz que a 9,2 m/s e earn=6 a vida deveria ser
/// `640/(1 - v·earn/64)` = 4654 ticks. O banco truncado é que impedia.
///
/// É a mesma armadilha do orçamento de 8.000 ticks: uma constante que ninguém conferiu
/// contra a tarefa vira o resultado.
static LIFE_CAP_TICKS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1920);
#[inline] fn life_cap() -> i64 { LIFE_CAP_TICKS.load(Ordering::Relaxed) }
/// Ticks de vida por metro avançado (lido dentro do laço, longe do parsing).
static LIFE_EARN: AtomicF32 = AtomicF32::new();

/// f32 atômico simples sobre AtomicU32 — só precisa de store/load, sem aritmética.
struct AtomicF32(std::sync::atomic::AtomicU32);
impl AtomicF32 {
    const fn new() -> Self { Self(std::sync::atomic::AtomicU32::new(0)) }
    fn store(&self, v: f32, o: Ordering) { self.0.store(v.to_bits(), o) }
    fn load(&self, o: Ordering) -> f32 { f32::from_bits(self.0.load(o)) }
}

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
    /// Teto de vida acumulável em ticks (0 = usar o padrão de 1920).
    let mut life_cap_arg: i64 = 0;
    let mut evo_n_elite:  usize = 3;
    let mut evo_tourney:  usize = 3;
    // gen_offset: número de geração inicial para relatório (útil ao retomar).
    let mut gen_offset:   usize = 0;
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
                "--seed"      => { i += 1; if i < args.len() { pop_seed      = args[i].parse().unwrap_or(0); } }
                "--wall-cost" => { i += 1; if i < args.len() { wall_cost     = args[i].parse().unwrap_or(0.0); } }
                "--brain-skip"=> { i += 1; if i < args.len() { brain_skip    = args[i].parse().unwrap_or(1).max(1); } }
                "--wall-life" => { i += 1; if i < args.len() { wall_life     = args[i].parse().unwrap_or(0); } }
                "--same-start"=> { same_start = true; }
                "--life-earn" => { i += 1; if i < args.len() { life_earn     = args[i].parse().unwrap_or(0.0); } }
                "--width"     => { i += 1; if i < args.len() { track_width   = args[i].parse().unwrap_or(18.0); } }
                "--fitness"   => { i += 1; if i < args.len() { modo_fitness = args[i].clone(); } }
                "--steer-alpha"=>{ i += 1; if i < args.len() { steer_alpha = args[i].parse().unwrap_or(0.5); } }
                "--life-cap"  => { i += 1; if i < args.len() { life_cap_arg  = args[i].parse().unwrap_or(0); } }
                "--n-elite"   => { i += 1; if i < args.len() { evo_n_elite   = args[i].parse().unwrap_or(3); } }
                "--tourney"   => { i += 1; if i < args.len() { evo_tourney   = args[i].parse().unwrap_or(3); } }
                "--gen-offset"=> { i += 1; if i < args.len() { gen_offset    = args[i].parse().unwrap_or(0); } }
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
    LIFE_EARN.store(life_earn, Ordering::Relaxed);
    STEER_ALPHA.store(steer_alpha.clamp(0.05, 1.0), Ordering::Relaxed);
    if life_cap_arg > 0 { LIFE_CAP_TICKS.store(life_cap_arg, Ordering::Relaxed); }
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

        // ── Reseta CfC entre gerações ──────────────────────────────────────────
        ctx.reset_cfc(n_seres);

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
                ticks_alive: 0,
                // Com vida ganha, o ser nasce com pouco fôlego (10 s) e precisa
                // conquistar o resto. Sem isso, ele teria a geração inteira garantida e o
                // mecanismo não teria efeito nenhum nos primeiros minutos.
                life_left: if LIFE_EARN.load(Ordering::Relaxed) > 0.0 { LIFE_START as f64 }
                           else { n_ticks as f64 },
                reverse_timer: 0.0,
                post_reverse_lockout: 0.0,
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
                    prog_acum[i] += d.clamp(-ld * 0.1, ld * 0.1) as f64;
                    prog_prev[i] = cur;

                    // Setor atual no circuito, 0..N_CHECKPOINTS-1.
                    let setor = ((cur / ld) * N_CHECKPOINTS as f32) as u32;
                    let esperado = checkpoint[i] % N_CHECKPOINTS;
                    if setor == esperado {
                        checkpoint[i] += 1;
                        if checkpoint[i] % N_CHECKPOINTS == 0 { voltas_reais[i] += 1; }
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

        // ── O campeão se REPRODUZ? ─────────────────────────────────────────────
        //
        // Pergunta do Leonardo: se o campeão sabe dar volta e a geração seguinte é quase
        // toda cópia mutada dele, o número de completadores tinha que CRESCER. Não cresce.
        //
        // `evolve()` passa os `n_elite` melhores SEM MUTAÇÃO. Logo `genomes[0]` desta
        // geração é literalmente o mesmo genoma que foi o melhor da anterior. Comparar a
        // fitness dele agora com a de então mede uma coisa só, e sem custo nenhum:
        // quanto do "campeão" era genoma e quanto era SORTE.
        //
        // Se o mesmo genoma marca 5000 m numa geração e 300 m na seguinte, a seleção está
        // escolhendo ruído, e nenhum filho pode herdar aquilo — porque não havia o que
        // herdar. Esse é o teto que nenhuma mutação melhor resolve.
        if let Some(anterior) = elite_fitness_anterior {
            let agora = dist_acum[0];
            let razao = if anterior > 1.0 { agora / anterior } else { f64::NAN };
            println!(
                "  [elite] mesmo genoma: {:.0} m → {:.0} m ({:.0}% do anterior)",
                anterior, agora, 100.0 * razao
            );
            elite_reprod.push(razao);
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
            _ => (0..n_seres).map(|i| {
                let setores = checkpoint[i] as f64 * (lap_dist / N_CHECKPOINTS as f64);
                setores + prog_acum[i].max(0.0) * 0.01
            }).collect(),
        };
        let r = gen_stats(gen_abs, &fitness, lap_dist);
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
        println!("  [setores] mediana {}/{} · melhor {}/{} · voltas reais: {} seres ≥1, {} ≥2",
                 cp_med, N_CHECKPOINTS, cp_max, N_CHECKPOINTS, v1, v2);

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
            let pr = universe::brain_capacity::participation_ratio(&samples);
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
    println!("=== FIM gpu_evolve ===");
}
