//! Observatório completo de uma corrida GPU — valide o que aconteceu sem depender de relatos.
//!
//! # Saídas
//!
//!   STDOUT         — mapa ASCII + tabela por-ser + perfis de velocidade ASCII + histogramas
//!   telemetry/gpu_run/summary.txt      — cópia completa do STDOUT
//!   telemetry/gpu_run/trajectories.csv — tick,ser,x,z,speed,throttle,steer,brake (cada SAMPLE_EVERY ticks)
//!   telemetry/gpu_run/accel_diag.txt   — aceleração medida vs teórica (ser 0, primeiros 200 ticks)
//!   telemetry/gpu_run/replay_<ts>.bin  — replay binário (posição, intent, sensores por tick amostrado)
//!
//! # Uso
//!
//!   cd ~/projects/LLM/Universe
//!   env -u DISPLAY ./target/release/gpu_report
//!   env -u DISPLAY ./target/release/gpu_report --seres 10 --ticks 3000
//!   env -u DISPLAY ./target/release/gpu_report --replay telemetry/gpu_run/replay_<ts>.bin --ser 0
//!
//! # Variáveis de ambiente
//!
//!   IARA_DIAG=1   — canais temporais do ser0 quando travado (a cada 200 ticks)
//!   IARA_ESCAPE=1 — habilita escape de estagnação (throttle=0.3 mínimo, interfere na pilotagem)
//!
//! # Regras de honestidade
//!
//!   - Nenhum número é impresso sem o dado que o sustenta.
//!   - Se todos os seres apresentam o MESMO valor numérico, o relatório DENUNCIA o fato:
//!     pode ser valor injetado/bug, não heterogeneidade real.
//!   - Seres com distância < 1 m são marcados como PARADO — não como "em movimento lento".
//!   - NÃO MEDIDO é escrito onde há lacuna real (ex.: arc-length quando racing_line vazia).
//!   - NUNCA imprime veredito sem o número que o sustenta.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use universe::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS};
use universe::config::iara::vision::{RETINA_ROWS, RETINA_TOTAL};
use universe::engine::SovereignEngine;
use universe::evolution::dna_genesis::load_champion_dna;
use universe::gpu::{Pose, RetinaGpu};
use universe::gpu::brain::{BrainGpu, BrainInput};
use universe::gpu::perception::{PerceptionGpu, PosePerception, TEMPORAL_STRIDE};
use universe::gpu::physics::{
    PhysicsGpu, PhysicsStateGpu, pack_config_flags, WheelGpu,
};
use universe::world::VoxelWorld;
use universe::simulation::racing_line::RacingLine;
use universe::visao::pitch;
use universe::perception::RetinaPerception;

// ─── Parâmetros de configuração ───────────────────────────────────────────────

/// Amostragem de trajetória: grava posição + pedais de TODOS os seres a cada N ticks.
const SAMPLE_EVERY: usize = 50;

/// Largura do mapa ASCII em colunas de terminal (76 = cabe em 80 colunas com margem).
const ASCII_COLS: usize = 76;

/// Seres cujo perfil de velocidade é impresso como gráfico ASCII.
const PERFIL_N_SERES: usize = 3;

/// Largura do gráfico ASCII de velocidade (colunas).
const PERFIL_COLS: usize = 60;

/// Altura do gráfico ASCII de velocidade (linhas).
const PERFIL_ROWS: usize = 12;

/// Número de bins no histograma de velocidade e distância.
const HIST_BINS: usize = 10;

/// Resolução do mapa de calor ASCII (células do mapa, não voxels).
/// Com grid 320 e ASCII_COLS=76, step=4 → 80 células.
const HEATMAP_COLS: usize = 76;

/// Intensidade mínima de hits para aparecer no mapa de calor
/// (filtra ruído de seres que passaram rapidamente).
const HEATMAP_MIN_HITS: usize = 2;

/// Ticks em torno do travamento que capturamos para análise neural.
/// Capturamos os últimos STUCK_WINDOW ticks antes do travamento.
const STUCK_WINDOW: usize = 5;

/// Limiar de saturação de um canal de intent: |v| > SAT_THRESH.
const SAT_THRESH: f32 = 0.90;

/// Número de ticks consecutivos com canal saturado para acionar alerta.
const SAT_MIN_TICKS: usize = 10;

// ─── Tipos locais (espelham gpu_loop.rs) ─────────────────────────────────────

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

/// Estado por-ser na CPU (física + controladores auxiliares).
struct BeingState {
    phys:                 PhysicsStateGpu,
    prev_steer:           f32,
    stagnation_ticks:     u32,
    escape_ticks:         u32,
    reverse_timer:        f32,
    post_reverse_lockout: f32,
}

fn pos_of(p: &PhysicsStateGpu) -> [f32; 3] { p.pos }

fn make_physics_state(start: [f32; 3], fwd: [f32; 3]) -> PhysicsStateGpu {
    // Carga normal estática = massa × g / 4 rodas (1350 kg como no gpu_loop.rs).
    let normal_load = 1350.0 * 9.81 / 4.0;
    let wheel = WheelGpu::from_cpu(0.0, 0.0, 0.0, normal_load, 60.0, 0.0, 0.0);
    let lat = [fwd[2], 0.0, -fwd[0]];
    PhysicsStateGpu {
        vel: [0.0; 3],
        av_y: 0.0,
        wheels: [wheel.part_a, wheel.part_b,
                 wheel.part_a, wheel.part_b,
                 wheel.part_a, wheel.part_b,
                 wheel.part_a, wheel.part_b],
        // Y=0 nasce abaixo do solo; a fase de colisão levanta para height_at+0.5.
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
        // Gear 3 = 2ª marcha (convenção: 0=ré, 1=neutro, 2=1ª, 3=2ª …).
        current_gear: 3,
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

// ─── GpuLoop ─────────────────────────────────────────────────────────────────
// Idêntico ao gpu_loop.rs — mantido aqui para que este binário seja autônomo
// e não dependa de alterações no arquivo principal enquanto outro agente trabalha.

struct GpuLoop {
    diag_feito:          bool,
    diag_fis:            bool,
    device:              Arc<wgpu::Device>,
    queue:               Arc<wgpu::Queue>,
    adapter_name:        String,
    pipeline_retina:     wgpu::ComputePipeline,
    layout_retina:       wgpu::BindGroupLayout,
    buf_grid:            wgpu::Buffer,
    buf_poses_retina:    wgpu::Buffer,
    buf_retina_out:      wgpu::Buffer,
    grid_size:           u32,
    cell_size:           f32,
    percept:             PerceptionGpu,
    brain:               BrainGpu,
    racing_line:         RacingLine,
    grid_cpu:            Vec<u32>,
    grid_size_cpu:       u32,
    cell_size_cpu:       f32,
    phys_gpu:            PhysicsGpu,
    kan_knots:           [[f32; universe::kan::KAN_OUT_KNOTS]; 3],
    percept_cfc:         Vec<RetinaPerception>,
    buf_retina_readback: wgpu::Buffer,
    capacity:            usize,
}

impl GpuLoop {
    fn new(
        world: &VoxelWorld,
        engines: &[SovereignEngine],
        capacity: usize,
        kan_knots: [[f32; universe::kan::KAN_OUT_KNOTS]; 3],
    ) -> Option<Self> {
        // Garante que apenas o ICD Radeon é carregado (múltiplos ICDs → SIGSEGV).
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
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            },
        ))?;
        let name = adapter.get_info().name.clone();
        let mut limits = wgpu::Limits::downlevel_defaults();
        // Cérebro usa 9 storage buffers (bindings 1-9); percepção usa 7 (max dos dois = 9).
        limits.max_storage_buffers_per_shader_stage = 9;
        let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label:             Some("iara-report"),
                required_features: wgpu::Features::empty(),
                required_limits:   limits,
                memory_hints:      wgpu::MemoryHints::Performance,
            },
            None,
        )).ok()?;
        let device = Arc::new(raw_device);
        let queue  = Arc::new(raw_queue);
        let cap = capacity.max(1);

        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("rep-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage:    wgpu::BufferUsages::STORAGE,
        });
        let buf_poses_retina = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("rep-poses-retina"),
            size:               (cap * std::mem::size_of::<PoseRaw>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_retina_out = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("rep-retina-out"),
            size:               (cap * RETINA_TOTAL * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("rep-retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                include_str!("../gpu/retina.wgsl"),
            )),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("rep-retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pl_retina = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&layout_retina], push_constant_ranges: &[],
        });
        let pipeline_retina = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("rep-retina-pipeline"),
            layout:              Some(&pl_retina),
            module:              &shader_retina,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        let percept = PerceptionGpu::new(&device, &queue, world, cap)?;
        let brain   = BrainGpu::new(Arc::clone(&device), Arc::clone(&queue), name.clone(), cap);
        let eng_refs: Vec<&SovereignEngine> = engines.iter().take(cap).collect();
        brain.upload_weights(&eng_refs);

        let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
        let phys_gpu = PhysicsGpu::new_shared(
            Arc::clone(&device), Arc::clone(&queue), name.clone(),
            &grid_u8, &world.height_map, world.size, world.cell_size, cap)?;

        let retina_rb_bytes = (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let buf_retina_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("rep-retina-readback"),
            size:               retina_rb_bytes,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let percept_cfc: Vec<RetinaPerception> = (0..cap).map(|_| RetinaPerception::new()).collect();

        Some(Self {
            diag_feito: false, diag_fis: false,
            device, queue, adapter_name: name,
            pipeline_retina, layout_retina,
            buf_grid, buf_poses_retina, buf_retina_out,
            grid_size: world.size as u32, cell_size: world.cell_size,
            percept, brain,
            racing_line: world.racing_line.clone(),
            grid_cpu: grid_u32, grid_size_cpu: world.size as u32, cell_size_cpu: world.cell_size,
            phys_gpu, kan_knots, percept_cfc, buf_retina_readback,
            capacity: cap,
        })
    }

    /// Um tick completo — idêntico ao GpuLoop::tick() de gpu_loop.rs.
    /// Retorna (ok, pedais[n]) onde pedais[i] = [throttle, steer, brake] final enviado à física.
    fn tick(
        &mut self,
        states:   &mut Vec<BeingState>,
        engines:  &[SovereignEngine],
        n:        usize,
        dt:       f32,
        tick_idx: usize,
    ) -> (bool, Vec<[f32; 3]>) {
        if n == 0 || n > self.capacity { return (false, vec![]); }

        // 1. Poses retina
        let pose_raw_data: Vec<PoseRaw> = (0..n).map(|i| {
            let p = &states[i].phys;
            PoseRaw {
                pos_gaze: [p.pos[0], p.pos[1], p.pos[2], 0.0],
                dirs:     [p.fwd[0], p.fwd[2], p.lat[0], p.lat[2]],
            }
        }).collect();
        self.queue.write_buffer(&self.buf_poses_retina, 0, bytemuck::cast_slice(&pose_raw_data));

        // 2. Poses percepção
        let poses_percept: Vec<PosePerception> = (0..n).map(|i| {
            let p = &states[i].phys;
            PosePerception {
                pos_x: p.pos[0], pos_y: p.pos[1], pos_z: p.pos[2], phase: 0.0,
                fwd_x: p.fwd[0], fwd_z: p.fwd[2], rgt_x: p.lat[0], rgt_z: p.lat[2],
            }
        }).collect();

        // 3a. Distâncias laterais e frontais (CPU) → temporal[16-18]
        {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            // Amostras laterais expandidas de 27m → 40m (conta: 40/28 m/s = 1.43 s antecipação).
            let lat_samp = [1.0f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 33.0, 40.0];
            let fwd_samp = [3.0f32, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0];
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };
            let mut dist_vals: Vec<[f32; 3]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let perp_x = -fz; let perp_z = fx;
                let mut dist_l = 40.0f32;  // default = livre (sem parede em 40m)
                for &sd in &lat_samp {
                    if voxel_at(px - perp_x * sd, pz - perp_z * sd) == 2u32 { dist_l = sd; break; }
                }
                let mut dist_r = 40.0f32;
                for &sd in &lat_samp {
                    if voxel_at(px + perp_x * sd, pz + perp_z * sd) == 2u32 { dist_r = sd; break; }
                }
                let mut dist_f = 50.0f32;
                for &sd in &fwd_samp {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2u32 || v == 3u32 { dist_f = sd; break; }
                }
                dist_vals.push([dist_l, dist_r, dist_f]);
            }
            self.percept.update_dist_temporal(&self.queue, &dist_vals[..n]);
        }

        // 3b. Racing line (CPU) → temporal[10-15]
        {
            let hw = self.racing_line.half_width.max(1.0);
            let mut racing_vals: Vec<[f32; 6]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let pos2d = bevy::math::Vec2::new(p.pos[0], p.pos[2]);
                let fwd2d = bevy::math::Vec2::new(p.fwd[0], p.fwd[2]);
                let line_off = self.racing_line.lateral_offset_from_line(pos2d, fwd2d);
                let off_norm = (line_off / hw).clamp(-1.0, 1.0);
                let la = self.racing_line.lookahead_offset(pos2d, 5.0);
                let la_norm = (la / hw).clamp(-1.0, 1.0);
                racing_vals.push([off_norm, 0.0, la_norm, 0.5, 0.8, 0.7]);
            }
            self.percept.update_racing_line_temporal(&self.queue, &racing_vals[..n]);
        }

        // 3c. Propriocepção → temporal[20-31]
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
            let mut prop_vals: Vec<[f32; 12]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed_m_s = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let vel_dot_fwd = p.vel[0] * p.fwd[0] + p.vel[2] * p.fwd[2];
                let alignment = if speed_m_s > 0.1 { (vel_dot_fwd / speed_m_s).clamp(-1.0, 1.0) } else { 1.0 };
                let lat_vel = p.vel[0] * p.lat[0] + p.vel[2] * p.lat[2];
                let rpm_ratio = if p.max_rpm > 0.0 { (p.current_rpm / p.max_rpm).clamp(0.0, 1.0) } else { 0.0 };
                let gear = p.current_gear as f32;
                let tire_temp_sum: f32 = (0..4).map(|j| p.wheels[j * 2 + 1][0]).sum();
                let tire_temp_norm = (tire_temp_sum / 4.0 / 120.0).clamp(0.0, 1.0);
                let voxel = voxel_at_cpu(p.pos[0], p.pos[2]);
                let on_grass = if voxel == 0u32 { 1.0f32 } else { 0.0 };
                prop_vals.push([
                    alignment, speed_m_s, lat_vel, 1.0,
                    rpm_ratio, gear, tire_temp_norm, 0.0,
                    0.0, on_grass, p.steer.abs().clamp(0.0, 1.0), 0.0,
                ]);
            }
            self.percept.update_proprioception_temporal(&self.queue, &prop_vals[..n]);
        }

        // 3d. reverse_timer + CfC percepção → temporal[32-48]
        {
            const REVERSE_MIN_DURATION: f32  = 0.4;
            const REVERSE_MAX_DURATION: f32  = 1.5;
            const REVERSE_EXIT_CLEAR: f32    = 2.5;
            const POST_REVERSE_LOCKOUT: f32  = 0.6;
            const STAGNATION_FOR_REVERSE: u32 = 60;
            const REV_SPEED_MAX: f32         = 1.5;
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32; let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };
            let mut extra_vals: Vec<[f32; 17]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let (px, pz) = (p.pos[0], p.pos[2]);
                let mut fwd_dist = 30.0f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2u32 || v == 3u32 { fwd_dist = sd; break; }
                }
                if states[i].post_reverse_lockout > 0.0 {
                    states[i].post_reverse_lockout = (states[i].post_reverse_lockout - dt).max(0.0);
                }
                {
                    let rev   = states[i].reverse_timer;
                    let stag  = states[i].stagnation_ticks;
                    let lock  = states[i].post_reverse_lockout;
                    if rev <= 0.0 && lock <= 0.0 && fwd_dist < 4.0
                        && speed < REV_SPEED_MAX && stag > STAGNATION_FOR_REVERSE
                    {
                        states[i].reverse_timer = REVERSE_MIN_DURATION;
                        states[i].stagnation_ticks = 0;
                    }
                }
                {
                    let rev = states[i].reverse_timer;
                    if rev > 0.0 {
                        let new_rev = rev - dt;
                        if fwd_dist > REVERSE_EXIT_CLEAR || new_rev > REVERSE_MAX_DURATION {
                            states[i].reverse_timer = 0.0;
                            states[i].post_reverse_lockout = POST_REVERSE_LOCKOUT;
                        } else {
                            states[i].reverse_timer = new_rev.max(0.0);
                        }
                    }
                }
                let hidden = &self.percept_cfc[i].pipeline.state.h;
                let mut val = [0.0f32; 17];
                val[0] = states[i].reverse_timer;
                for j in 0..16 { val[1 + j] = if j < hidden.len() { hidden[j] } else { 0.0 }; }
                extra_vals.push(val);
            }
            self.percept.update_extra_temporal(&self.queue, &extra_vals[..n]);
        }

        // 4. Encoder único (retina → percepção → cérebro)
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("rep-tick") });

        // 4a. Retina
        let mut pitches_arr = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS { pitches_arr[row / 4][row % 4] = pitch(row); }
        let retina_params = RetinaParamsGpu {
            grid_size: self.grid_size, n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half:  (universe::visao::fov_deg() / 2.0).to_radians(),
            pitches:   pitches_arr,
        };
        let buf_rp = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("rep-retina-params"),
            contents: bytemuck::bytes_of(&retina_params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let bg_retina = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layout_retina,
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
                label: Some("rep-pass-retina"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }

        // 4b. Percepção
        self.percept.bind_and_dispatch(
            &mut enc, &self.device, &self.queue, &poses_percept[..n], &self.buf_retina_out);

        // 4c. Cérebro
        self.brain.bind_and_dispatch(
            &mut enc, n, &[], dt, Some(self.percept.buf_output()));

        // 4d. Cópia retina → readback (para CfC de percepção na CPU)
        let retina_copy_bytes = (n * RETINA_TOTAL * 4) as u64;
        enc.copy_buffer_to_buffer(
            &self.buf_retina_out, 0, &self.buf_retina_readback, 0, retina_copy_bytes);

        // 4e. Submit único
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // 5. Leitura do intent
        let intents = self.brain.read_intent_after_submit(n);

        // 6. Readback retina → CfC percepção
        {
            let slice = self.buf_retina_readback.slice(0..retina_copy_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();
            let retina_data = slice.get_mapped_range();
            let retina_u32: &[u32] = bytemuck::cast_slice(&retina_data);
            for i in 0..n {
                let base = i * RETINA_TOTAL;
                let mut retina_i8 = vec![0i8; RETINA_TOTAL];
                for j in 0..RETINA_TOTAL {
                    let packed  = retina_u32[base + j];
                    let dist_mm = (packed >> 8) as f32;
                    let dist_m  = dist_mm / 1000.0;
                    let mat     = packed & 0xFF;
                    let v: f32  = match mat {
                        2 => if dist_m <= 30.0 { 127.0 - (127.0 - 55.0) * (dist_m / 30.0) }
                             else { 55.0 * (1.0 - (dist_m - 30.0) / 90.0).max(0.0) },
                        1 => -128.0 + (dist_m.min(80.0) / 80.0) * 128.0,
                        0 => 40.0,
                        _ => 127.0,
                    };
                    retina_i8[j] = v.clamp(-127.0, 127.0) as i8;
                }
                self.percept_cfc[i].perceive(&retina_i8);
            }
            drop(retina_data);
            self.buf_retina_readback.unmap();
        }

        if intents.len() != n { return (false, vec![]); }

        // 7. Aplica intent + física
        const REVERSE_PEDAL: f32 = 0.75;
        let kan_eval_fn = |out_idx: usize, x: f32| -> f32 {
            let xc   = x.clamp(-2.0, 2.0);
            let u    = (xc + 2.0) / 4.0;
            let fpos = u * 11.0;
            let i0   = fpos.floor() as usize;
            let i1   = (i0 + 1).min(11);
            let frac = fpos - (i0 as f32);
            let a    = self.kan_knots[out_idx][i0];
            let b    = self.kan_knots[out_idx][i1];
            (a + (b - a) * frac).clamp(-1.0, 1.0)
        };

        let mut pedals: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut phys_states: Vec<PhysicsStateGpu> = (0..n).map(|i| {
            let mut s       = states[i].phys;
            let intent      = &intents[i];
            let thr_kan     = kan_eval_fn(0, intent[0]).max(0.0);
            let steer_kan   = kan_eval_fn(1, intent[1]);
            let brk_kan     = kan_eval_fn(2, intent[2]).max(0.0);
            let steer_smooth = (states[i].prev_steer * 0.50 + steer_kan * 0.50).clamp(-1.0, 1.0);
            let is_reversing = states[i].reverse_timer > 0.0;
            let in_lockout   = states[i].post_reverse_lockout > 0.0;
            let speed_now    = (s.vel[0].powi(2) + s.vel[2].powi(2)).sqrt();
            // Anula brake quando speed < 2 m/s (evita travar carro parado — mesmo que motor.rs).
            let brk_final   = if speed_now < 2.0 { 0.0 } else { brk_kan };
            if is_reversing {
                s.throttle = REVERSE_PEDAL; s.steer = 0.0; s.brake_cmd = 0.0; s.current_gear = 0;
            } else if in_lockout {
                s.throttle = thr_kan.min(0.8); s.steer = steer_smooth; s.brake_cmd = brk_final;
                if s.current_gear <= 1 { s.current_gear = 2; }
            } else if states[i].escape_ticks > 0 {
                s.throttle = 0.3f32.max(thr_kan); s.steer = steer_smooth; s.brake_cmd = 0.0;
                if s.current_gear <= 1 { s.current_gear = 2; }
            } else {
                s.throttle = thr_kan; s.steer = steer_smooth; s.brake_cmd = brk_final;
                if s.current_gear <= 1 { s.current_gear = 2; }
                if s.current_gear > 2 && speed_now < 3.0 { s.current_gear -= 1; }
            }
            s.dt = dt;
            pedals.push([s.throttle, s.steer, s.brake_cmd]);
            s
        }).collect();

        for i in 0..n {
            if states[i].reverse_timer > 0.0 { states[i].prev_steer = 0.0; }
            else { states[i].prev_steer = phys_states[i].steer; }
            let speed = (phys_states[i].vel[0].powi(2) + phys_states[i].vel[2].powi(2)).sqrt();
            if speed < 1.5 { states[i].stagnation_ticks += 1; }
            else { states[i].stagnation_ticks = 0; }
            if std::env::var("IARA_ESCAPE").is_ok() {
                if states[i].stagnation_ticks >= 32 && states[i].escape_ticks == 0 {
                    states[i].escape_ticks = 64; states[i].stagnation_ticks = 0;
                }
            }
            if states[i].escape_ticks > 0 { states[i].escape_ticks -= 1; }
        }

        self.phys_gpu.step(&mut phys_states);
        for i in 0..n { states[i].phys = phys_states[i]; }

        let memory0_vals: Vec<f32> = (0..n).map(|i| intents[i][0].clamp(0.0, 1.0)).collect();
        self.percept.update_memory0(&self.queue, &memory0_vals[..n]);

        (true, pedals)
    }
}

// ─── Render ASCII do mapa da pista ────────────────────────────────────────────

/// Símbolos: `#` = wall, `.` = road, ` ` = grass, `S` = pos inicial, `F` = pos final, `T` = travado
fn render_ascii_track(
    world:    &VoxelWorld,
    pos_ini:  &[[f32; 3]],
    pos_fin:  &[[f32; 3]],
    stuck:    &[[f32; 3]],   // posições onde seres ficaram travados
    cols:     usize,
) -> Vec<String> {
    let gs   = world.size;       // 320
    let step = (gs / cols).max(1);
    let cols_real = gs / step;
    let rows_real = gs / step;

    // Tipo dominante por célula display (wall > road > grass).
    let mut grid_disp = vec![0u8; cols_real * rows_real];
    for row in 0..rows_real {
        for col in 0..cols_real {
            let mut n_wall = 0usize; let mut n_road = 0usize;
            for dr in 0..step { for dc in 0..step {
                let wr = row * step + dr; let wc = col * step + dc;
                if wr < gs && wc < gs {
                    match world.grid[wr * gs + wc] as u8 {
                        2 => n_wall += 1, 1 => n_road += 1, _ => {}
                    }
                }
            }}
            grid_disp[row * cols_real + col] =
                if n_wall > 0 { 2 } else if n_road > 0 { 1 } else { 0 };
        }
    }

    // Marcadores: T (travado) sobrescrito por S (início) sobrescrito por F (fim).
    // Ordem de prioridade: F > S > T (mais informativo no display).
    let mut markers: std::collections::HashMap<(usize, usize), char> = Default::default();

    // Converte coordenada mundo → célula display
    let world_to_disp = |wx: f32, wz: f32| -> (usize, usize) {
        let col = ((wx / world.cell_size) as usize / step).min(cols_real.saturating_sub(1));
        let row = ((wz / world.cell_size) as usize / step).min(rows_real.saturating_sub(1));
        (row, col)
    };

    for p in stuck  { let (r, c) = world_to_disp(p[0], p[2]); markers.entry((r, c)).or_insert('T'); }
    for p in pos_ini { let (r, c) = world_to_disp(p[0], p[2]); markers.entry((r, c)).or_insert('S'); }
    for p in pos_fin { let (r, c) = world_to_disp(p[0], p[2]); markers.insert((r, c), 'F'); }

    let mut lines = Vec::with_capacity(rows_real);
    for row in 0..rows_real {
        let mut line = String::with_capacity(cols_real);
        for col in 0..cols_real {
            line.push(if let Some(&mk) = markers.get(&(row, col)) {
                mk
            } else {
                match grid_disp[row * cols_real + col] {
                    2 => '#', 1 => '.', _ => ' ',
                }
            });
        }
        lines.push(line);
    }
    lines
}

// ─── Arc-length na racing_line ────────────────────────────────────────────────

/// Arc-length do ponto mais próximo da racing_line à posição `pos`.
/// Retorna -1.0 quando a racing_line está vazia (NÃO MEDIDO).
fn arc_progress(rl: &RacingLine, pos: [f32; 3]) -> f32 {
    if rl.points.is_empty() { return -1.0; }
    let pos2d = bevy::math::Vec2::new(pos[0], pos[2]);
    let mut best_d   = f32::INFINITY;
    let mut best_arc = 0.0f32;
    for (j, &pt) in rl.points.iter().enumerate() {
        let d = pos2d.distance_squared(pt);
        if d < best_d { best_d = d; best_arc = rl.arc[j]; }
    }
    best_arc
}

// ─── Gráfico ASCII de velocidade × tick ───────────────────────────────────────

/// Renderiza série temporal de velocidade `vals` (m/s) como gráfico ASCII.
/// `rows` = altura do gráfico; `cols` = largura.
/// Retorna vetor de strings (linhas).
fn ascii_velocity_chart(vals: &[(usize, f32)], rows: usize, cols: usize, titulo: &str) -> Vec<String> {
    if vals.is_empty() { return vec![format!("[{} — sem dados]", titulo)]; }

    let v_max = vals.iter().map(|&(_, v)| v).fold(0.0f32, f32::max);
    let v_max = if v_max < 0.01 { 1.0 } else { v_max };

    // Comprime série temporal para `cols` colunas (média em cada bloco).
    let n    = vals.len();
    let step = (n / cols).max(1);
    let n_cols = n / step;
    let mut compressed: Vec<f32> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let start = c * step;
        let end   = ((c + 1) * step).min(n);
        let avg   = vals[start..end].iter().map(|&(_, v)| v).sum::<f32>() / (end - start) as f32;
        compressed.push(avg);
    }

    // Constrói grade de caracteres: rows linhas × n_cols colunas.
    let mut grid = vec![vec![' '; n_cols]; rows];
    for (c, &v) in compressed.iter().enumerate() {
        // Quantas linhas a preencher de baixo para cima.
        let filled = ((v / v_max) * (rows as f32)).round() as usize;
        let filled = filled.min(rows);
        for r in (rows - filled)..rows {
            grid[r][c] = '|';
        }
    }

    let mut lines = Vec::with_capacity(rows + 4);
    // Cabeçalho com título
    lines.push(format!("  {} (max={:.1} m/s = {:.0} km/h)", titulo, v_max, v_max * 3.6));
    // Eixo Y (topo = v_max, base = 0)
    for (r, row) in grid.iter().enumerate() {
        let y_val = v_max * (1.0 - r as f32 / rows as f32);
        let label = if r == 0 || r == rows / 2 || r == rows - 1 {
            format!("{:5.1}|", y_val)
        } else {
            "     |".to_string()
        };
        let row_str: String = row.iter().collect();
        lines.push(format!("{}{}", label, row_str));
    }
    // Eixo X
    lines.push(format!("     +{}", "-".repeat(n_cols)));
    let t_max = vals.last().map(|&(t, _)| t).unwrap_or(0);
    lines.push(format!("      tick 0{:>width$}", t_max, width = n_cols.saturating_sub(6)));
    lines
}

// ─── Histograma ASCII ─────────────────────────────────────────────────────────

/// Histograma de N bins impresso como barras horizontais.
fn ascii_histogram(vals: &[f64], bins: usize, label: &str) -> Vec<String> {
    if vals.is_empty() { return vec![format!("[{}: sem dados]", label)]; }
    let v_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let v_max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (v_max - v_min).abs() < 1e-9 {
        // Todos iguais — sinal de bug.
        return vec![
            format!("  {}: TODOS OS VALORES IGUAIS ({:.3})", label, v_min),
            format!("  ⚠ Se isso for inesperado, suspeite de valor injetado ou bug."),
        ];
    }
    let range    = v_max - v_min;
    let bin_size = range / bins as f64;
    let mut counts = vec![0usize; bins];
    for &v in vals {
        let b = ((v - v_min) / bin_size).floor() as usize;
        counts[b.min(bins - 1)] += 1;
    }
    let count_max = *counts.iter().max().unwrap_or(&1).max(&1);
    let bar_width = 30;

    let mut lines = Vec::new();
    lines.push(format!("  {} (N={}, min={:.1}, max={:.1})", label, vals.len(), v_min, v_max));
    for (b, &cnt) in counts.iter().enumerate() {
        let lo   = v_min + b as f64 * bin_size;
        let hi   = lo + bin_size;
        let fill = (cnt * bar_width / count_max).max(if cnt > 0 { 1 } else { 0 });
        let bar  = "#".repeat(fill);
        lines.push(format!("  [{:7.1},{:7.1}) {:3}  {}", lo, hi, cnt, bar));
    }
    lines
}

// ─── Detecção de homogeneidade suspeita ───────────────────────────────────────

/// Retorna `Some(msg)` se todos os valores em `vals` são iguais (bug suspeito).
fn check_homogeneidade(vals: &[f64], nome: &str) -> Option<String> {
    if vals.len() < 2 { return None; }
    let first = vals[0];
    let all_same = vals.iter().all(|&v| (v - first).abs() < 1e-6);
    if all_same {
        Some(format!(
            "⚠ ALERTA: todos os {} seres têm {}={:.6} — suspeite de valor injetado ou bug de pilotagem!",
            vals.len(), nome, first
        ))
    } else {
        None
    }
}

// ─── Mapa de calor ASCII ──────────────────────────────────────────────────────

/// Renderiza mapa de calor sobreposto à pista ASCII.
/// `heat_grid[row][col]` = número de frames de seres naquela célula display.
/// Intensidade: espaço=0, `.`=1-4, `+`=5-9, `o`=10-19, `O`=20-49, `*`=50-99, `@`=>=100.
fn render_heatmap(
    world:     &VoxelWorld,
    heat_grid: &[Vec<usize>],
    cols:      usize,
    stuck_positions: &[[f32; 3]],
) -> Vec<String> {
    let gs   = world.size;
    let step = (gs / cols).max(1);
    let cols_real = gs / step;
    let rows_real = gs / step;

    // Tipo dominante por célula display (wall > road > grass).
    let mut grid_disp = vec![0u8; cols_real * rows_real];
    for row in 0..rows_real {
        for col in 0..cols_real {
            let mut n_wall = 0usize; let mut n_road = 0usize;
            for dr in 0..step { for dc in 0..step {
                let wr = row * step + dr; let wc = col * step + dc;
                if wr < gs && wc < gs {
                    match world.grid[wr * gs + wc] as u8 {
                        2 => n_wall += 1, 1 => n_road += 1, _ => {}
                    }
                }
            }}
            grid_disp[row * cols_real + col] =
                if n_wall > 0 { 2 } else if n_road > 0 { 1 } else { 0 };
        }
    }

    // Marcadores de posição de travamento (prioridade máxima)
    let world_to_disp = |wx: f32, wz: f32| -> (usize, usize) {
        let col = ((wx / world.cell_size) as usize / step).min(cols_real.saturating_sub(1));
        let row = ((wz / world.cell_size) as usize / step).min(rows_real.saturating_sub(1));
        (row, col)
    };
    let mut stuck_marks: std::collections::HashSet<(usize, usize)> = Default::default();
    for p in stuck_positions {
        let (r, c) = world_to_disp(p[0], p[2]);
        stuck_marks.insert((r, c));
    }

    let mut lines = Vec::with_capacity(rows_real + 4);
    lines.push(format!(
        "  Intensidade: '.'=1-4  '+'=5-9  'o'=10-19  'O'=20-49  '*'=50-99  '@'=>=100  'X'=travamento"
    ));
    lines.push(format!(
        "  (fundo: '#'=wall  espaço=grass  células sem tráfego = road vazia)"
    ));

    for row in 0..rows_real {
        let mut line = String::with_capacity(cols_real);
        for col in 0..cols_real {
            let hits = if row < heat_grid.len() && col < heat_grid[row].len() {
                heat_grid[row][col]
            } else { 0 };

            // Sinal de travamento tem prioridade
            if stuck_marks.contains(&(row, col)) {
                line.push('X');
                continue;
            }

            let base_char = match grid_disp[row * cols_real + col] {
                2 => '#',
                1 => {
                    if hits < HEATMAP_MIN_HITS { '.' }
                    else if hits < 5  { '.' }
                    else if hits < 10 { '+' }
                    else if hits < 20 { 'o' }
                    else if hits < 50 { 'O' }
                    else if hits < 100 { '*' }
                    else { '@' }
                }
                _ => {
                    if hits >= HEATMAP_MIN_HITS { '+' } else { ' ' }
                }
            };
            line.push(base_char);
        }
        lines.push(line);
    }
    lines
}

/// Acumula posições de todos os seres em cada tick para o mapa de calor.
/// Retorna grid[row][col] = contagem de frames.
fn build_heatmap(
    world:  &VoxelWorld,
    frames: &[(f32, f32)],  // (x, z) de cada frame amostrado
    cols:   usize,
) -> Vec<Vec<usize>> {
    let gs   = world.size;
    let step = (gs / cols).max(1);
    let cols_real = gs / step;
    let rows_real = gs / step;
    let mut grid = vec![vec![0usize; cols_real]; rows_real];

    for &(wx, wz) in frames {
        let col = ((wx / world.cell_size) as usize / step).min(cols_real.saturating_sub(1));
        let row = ((wz / world.cell_size) as usize / step).min(rows_real.saturating_sub(1));
        grid[row][col] += 1;
    }
    grid
}

// ─── Análise de estado neural no instante de falha ───────────────────────────

/// Snapshot de intent+sensores no instante de travamento.
struct StuckBrainSnapshot {
    ser:    usize,
    #[allow(dead_code)]
    tick:   usize,
    intent: Vec<f32>,
    dist_l: f32,
    dist_r: f32,
    dist_f: f32,
}

/// Analisa snapshots de intent no instante de travamento.
/// Retorna relatório textual.
fn analyze_stuck_brain(
    snapshots: &[StuckBrainSnapshot],
    n_canais: usize,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    if snapshots.is_empty() {
        lines.push("  Nenhum snapshot de travamento capturado.".to_string());
        return lines;
    }

    let n = snapshots.len();
    lines.push(format!("  Snapshots capturados: {} (um por ser travado)", n));

    // 1. Dimensões saturadas por snapshot
    let mut total_sat_dims = 0usize;
    let mut sat_counts_per_dim = vec![0usize; n_canais];

    for snap in snapshots {
        let sat: Vec<usize> = snap.intent.iter().enumerate()
            .filter(|&(_, &v)| v.abs() > SAT_THRESH)
            .map(|(i, _)| i)
            .collect();
        total_sat_dims += sat.len();
        for &ch in &sat {
            if ch < n_canais { sat_counts_per_dim[ch] += 1; }
        }
    }

    let avg_sat = total_sat_dims as f32 / n as f32;
    lines.push(format!(
        "  Dims saturadas (|v|>{:.2}) por ser no travamento: {:.1} em media ({} total em {} seres x {} dims)",
        SAT_THRESH, avg_sat, total_sat_dims, n, n_canais
    ));

    // Canais saturados em TODOS os seres (sinal de ponto fixo / valor injetado)
    let mut universal: Vec<usize> = Vec::new();
    let mut majoritario: Vec<usize> = Vec::new();
    for ch in 0..n_canais {
        let cnt = sat_counts_per_dim[ch];
        if cnt == n && n > 1 { universal.push(ch); }
        else if cnt > n / 2 && n > 1 { majoritario.push(ch); }
    }

    if !universal.is_empty() {
        lines.push(format!(
            "  ALERTA PONTO FIXO: canais saturados em TODOS os {} seres: {:?}",
            n, universal
        ));
        lines.push("    -> Indicio de ponto fixo ou valor injetado — cerebro parou de variar.".to_string());
    } else if !majoritario.is_empty() {
        lines.push(format!(
            "  ALERTA: Canais saturados em >50% dos seres ({}/{}): {:?}",
            n, n, majoritario
        ));
    } else {
        lines.push(format!(
            "  OK: Sem canal universalmente saturado — heterogeneidade confirmada."
        ));
    }

    // 2. Estado estático? Compara intent entre snapshots
    if n >= 2 {
        let mut max_diff = 0.0f32;
        for ch in 0..n_canais {
            let vals: Vec<f32> = snapshots.iter()
                .filter_map(|s| s.intent.get(ch).cloned())
                .collect();
            if vals.len() >= 2 {
                let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let diff = mx - mn;
                if diff > max_diff { max_diff = diff; }
            }
        }
        if max_diff < 0.01 {
            lines.push(format!(
                "  ALERTA ESTADO IDENTICO ENTRE SERES (max_diff={:.4}): possivel ponto fixo global.",
                max_diff
            ));
        } else {
            lines.push(format!(
                "  OK: Intent varia entre seres (max_diff entre canais={:.3}) — estados distintos.",
                max_diff
            ));
        }
    }

    // 3. Dados sensoriais no travamento: VE a parede ou nao?
    lines.push(String::new());
    lines.push("  === CANAIS SENSORIAIS NO TRAVAMENTO ===".to_string());
    lines.push(format!(
        "  {:>4}  {:>6}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}",
        "ser", "dist_l", "dist_r", "dist_f", "int[0]", "int[1]", "int[2]"
    ));

    let mut n_ve_parede  = 0usize;
    let mut n_reage      = 0usize;
    let mut n_nao_ve     = 0usize;
    let mut n_ve_nao_reage = 0usize;

    for snap in snapshots {
        let thr = snap.intent.get(0).cloned().unwrap_or(0.0);
        let str_ = snap.intent.get(1).cloned().unwrap_or(0.0);
        let brk = snap.intent.get(2).cloned().unwrap_or(0.0);
        lines.push(format!(
            "  {:>4}  {:>6.1}  {:>6.1}  {:>6.1}  {:>8.3}  {:>8.3}  {:>8.3}",
            snap.ser, snap.dist_l, snap.dist_r, snap.dist_f, thr, str_, brk
        ));

        if snap.dist_f < 5.0 {
            n_ve_parede += 1;
            if brk > 0.3 || thr < 0.3 || str_.abs() > 0.5 {
                n_reage += 1;
            } else {
                n_ve_nao_reage += 1;
            }
        } else {
            n_nao_ve += 1;
        }
    }

    lines.push(String::new());
    lines.push(format!("  === VEREDITO SENSORIAL ({} seres) ===", n));
    lines.push(format!(
        "  Ve parede (dist_f<5m): {}/{} ({:.0}%)",
        n_ve_parede, n, 100.0 * n_ve_parede as f32 / n as f32
    ));
    if n_ve_parede > 0 {
        lines.push(format!(
            "    -> Reage (freia/vira/solta gas): {}/{} ({:.0}%)  — NAO reage: {}/{}",
            n_reage, n_ve_parede,
            100.0 * n_reage as f32 / n_ve_parede as f32,
            n_ve_nao_reage, n_ve_parede
        ));
    }
    lines.push(format!(
        "  NAO ve parede (dist_f>=5m mas travado): {}/{} ({:.0}%)",
        n_nao_ve, n, 100.0 * n_nao_ve as f32 / n as f32
    ));

    // Veredito final
    if n_ve_parede > 0 && n_ve_nao_reage > n_reage {
        lines.push(format!(
            "  *** DIAGNOSTICO: ser VE a parede ({} casos) mas NAO reage ({} nao-reage vs {} reage) -> falha no controle/intent. ***",
            n_ve_parede, n_ve_nao_reage, n_reage
        ));
    } else if n_nao_ve > n_ve_parede {
        lines.push(format!(
            "  *** DIAGNOSTICO: ser NAO VE a parede ({} vs {} que ve) -> range sensor limitado ou esterco satura antes. ***",
            n_nao_ve, n_ve_parede
        ));
    } else if n_ve_parede > 0 && n_reage >= n_ve_nao_reage {
        lines.push(format!(
            "  *** DIAGNOSTICO: ser ve ({}) E reage ({}) mas trava — possivel inercia/fisica. ***",
            n_ve_parede, n_reage
        ));
    } else {
        lines.push("  *** DIAGNOSTICO: padrao misto — analisar individualmente acima. ***".to_string());
    }

    lines
}

// ─── Acumulador de saturacao persistente ─────────────────────────────────────

struct SatTracker {
    sat_ticks: Vec<Vec<usize>>,
    alertas:   Vec<String>,
}

impl SatTracker {
    fn new(n_seres: usize, n_canais: usize) -> Self {
        Self {
            sat_ticks: vec![vec![0usize; n_canais]; n_seres],
            alertas:   Vec::new(),
        }
    }

    fn update(&mut self, ser: usize, tick: usize, intent: &[f32]) {
        if ser >= self.sat_ticks.len() { return; }
        for (ch, &v) in intent.iter().enumerate() {
            if ch >= self.sat_ticks[ser].len() { break; }
            if v.abs() > SAT_THRESH {
                self.sat_ticks[ser][ch] += 1;
                if self.sat_ticks[ser][ch] == SAT_MIN_TICKS {
                    let alerta = format!(
                        "  SATURACAO: ser[{}] intent[{}]={:+.3} saturado por >={} ticks consecutivos (tick {})",
                        ser, ch, v, SAT_MIN_TICKS, tick
                    );
                    self.alertas.push(alerta);
                }
            } else {
                self.sat_ticks[ser][ch] = 0;
            }
        }
    }
}

// ─── Auto-diagnostico ─────────────────────────────────────────────────────────

fn auto_diagnostico(
    stuck_positions:      &[Option<[f32; 3]>],
    n_seres:              usize,
    world:                &VoxelWorld,
    all_intents_at_stuck: &[Option<Vec<f32>>],
    all_dist_at_stuck:    &[Option<[f32; 3]>],
    cols:                 usize,
) -> Vec<String> {
    let mut diags: Vec<String> = Vec::new();

    let n_travados: usize = stuck_positions.iter().filter(|p| p.is_some()).count();
    if n_travados == 0 {
        diags.push("  OK: Nenhum ser travou (>100 ticks parado).".to_string());
        return diags;
    }

    // 1. Cluster de travamento: gargalo de pista vs falha cerebral distribuida
    let gs   = world.size;
    let step = (gs / cols).max(1);
    let cols_real = gs / step;
    let world_to_cell = |wx: f32, wz: f32| -> (usize, usize) {
        let c = ((wx / world.cell_size) as usize / step).min(cols_real.saturating_sub(1));
        let r = ((wz / world.cell_size) as usize / step).min(cols_real.saturating_sub(1));
        (r, c)
    };

    let mut cell_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for p in stuck_positions.iter().flatten() {
        *cell_counts.entry(world_to_cell(p[0], p[2])).or_insert(0) += 1;
    }

    let n_cells  = cell_counts.len();
    let max_cell = cell_counts.values().cloned().max().unwrap_or(0);
    let frac_gargalo = max_cell as f32 / n_travados.max(1) as f32;

    diags.push(format!(
        "  Travamentos: {}/{} seres em {} celulas distintas (max_densidade={} seres/celula, {:.0}% no pior ponto)",
        n_travados, n_seres, n_cells, max_cell, 100.0 * frac_gargalo
    ));

    if frac_gargalo > 0.5 {
        diags.push(format!(
            "  GARGALO DE PISTA: {:.0}% dos seres travaram na mesma regiao — curva especifica ou parede pontual.",
            100.0 * frac_gargalo
        ));
    } else if n_cells >= n_travados.saturating_sub(2) {
        diags.push(format!(
            "  FALHA CEREBRAL DISTRIBUIDA: {} seres em {} celulas diferentes — nao e um ponto fixo da pista.",
            n_travados, n_cells
        ));
    } else {
        diags.push(format!(
            "  Padrao misto: {} pontos de travamento para {} seres.",
            n_cells, n_travados
        ));
    }

    // 2. Intent identicos entre seres? (valor injetado)
    let intents_validos: Vec<&Vec<f32>> = all_intents_at_stuck.iter()
        .filter_map(|o| o.as_ref())
        .collect();

    if intents_validos.len() >= 2 {
        let n_ch = intents_validos[0].len();
        let mut n_canais_identicos = 0usize;
        for ch in 0..n_ch {
            let vals: Vec<f32> = intents_validos.iter()
                .filter_map(|iv| iv.get(ch).cloned())
                .collect();
            if vals.len() >= 2 {
                let range = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    - vals.iter().cloned().fold(f32::INFINITY, f32::min);
                if range < 1e-4 { n_canais_identicos += 1; }
            }
        }
        let frac_id = n_canais_identicos as f32 / n_ch as f32;
        if frac_id > 0.8 {
            diags.push(format!(
                "  ALERTA VALORES IDENTICOS: {}/{} canais de intent IGUAIS entre seres no travamento ({:.0}%) — suspeite de valor injetado.",
                n_canais_identicos, n_ch, 100.0 * frac_id
            ));
        } else {
            diags.push(format!(
                "  OK: Intent diverso no travamento: {}/{} canais diferem ({:.0}% variando).",
                n_ch - n_canais_identicos, n_ch, 100.0 * (1.0 - frac_id)
            ));
        }
    }

    // 3. Dist frontal media no travamento
    let dists_f: Vec<f32> = all_dist_at_stuck.iter()
        .filter_map(|o| *o)
        .map(|[_, _, df]| df)
        .collect();
    if !dists_f.is_empty() {
        let avg_df = dists_f.iter().sum::<f32>() / dists_f.len() as f32;
        let perto = dists_f.iter().filter(|&&d| d < 5.0).count();
        let longe = dists_f.iter().filter(|&&d| d >= 5.0).count();
        diags.push(format!(
            "  Dist frontal no travamento: avg={:.1}m  perto(<5m)={}  longe(>=5m)={}",
            avg_df, perto, longe
        ));
        if perto > longe {
            diags.push("  -> Maioria PERTO da parede — parede visivel. Ver VEREDITO SENSORIAL.".to_string());
        } else {
            diags.push("  -> Maioria LONGE da parede — possivel travamento em trecho aberto (velocidade nula).".to_string());
        }
    } else {
        diags.push("  Dist frontal no travamento: NAO MEDIDO (snapshots nao disponiveis).".to_string());
    }

    diags
}

// ─── Replay: escrita e leitura ────────────────────────────────────────────────

/// Frame de replay por ser por tick amostrado.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReplayFrame {
    tick:      u32,
    ser:       u32,
    x:         f32,
    z:         f32,
    speed:     f32,
    throttle:  f32,
    steer:     f32,
    brake:     f32,
    dist_l:    f32,
    dist_r:    f32,
    dist_f:    f32,
    fwd_x:     f32,
    fwd_z:     f32,
    intent_0:  f32,
    intent_1:  f32,
    intent_2:  f32,
    _pad:      f32,
}

/// Cabecalho do arquivo de replay.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReplayHeader {
    magic:        u32,
    version:      u32,
    n_seres:      u32,
    n_ticks:      u32,
    sample_every: u32,
    seed:         u32,
    _pad:         [u32; 2],
}

const REPLAY_MAGIC:   u32 = 0x52504C59; // "RPLY"
const REPLAY_VERSION: u32 = 1;
const REPLAY_SAMPLE_EVERY: usize = 50;

fn save_replay(path: &str, n_seres: usize, n_ticks: usize, frames: &[ReplayFrame]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    let header = ReplayHeader {
        magic: REPLAY_MAGIC, version: REPLAY_VERSION,
        n_seres: n_seres as u32, n_ticks: n_ticks as u32,
        sample_every: REPLAY_SAMPLE_EVERY as u32, seed: 9001,
        _pad: [0, 0],
    };
    f.write_all(bytemuck::bytes_of(&header))?;
    f.write_all(bytemuck::cast_slice(frames))?;
    Ok(())
}

fn load_and_print_replay(path: &str, ser_target: usize) {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => { eprintln!("Erro ao ler {}: {}", path, e); return; }
    };
    let hdr_size = std::mem::size_of::<ReplayHeader>();
    if data.len() < hdr_size { eprintln!("Arquivo de replay muito pequeno."); return; }
    let header: ReplayHeader = *bytemuck::from_bytes(&data[..hdr_size]);
    if header.magic != REPLAY_MAGIC {
        eprintln!("Magic invalido: {:08x}", header.magic); return;
    }

    println!("=== REPLAY: {} ===", path);
    println!("versao={} n_seres={} n_ticks={} sample_every={} seed={}",
        header.version, header.n_seres, header.n_ticks, header.sample_every, header.seed);
    println!("Exibindo ser={}", ser_target);
    println!();

    let frame_size = std::mem::size_of::<ReplayFrame>();
    let body = &data[hdr_size..];
    let n_frames = body.len() / frame_size;

    println!("{:>6}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7}  {:>6}  {:>6}  {:>6}  {:>7}  {:>7}  {:>7}",
        "tick", "x", "z", "speed", "thr", "steer", "brake",
        "d_esq", "d_dir", "d_frt", "int0", "int1");

    let mut prev_x = 0.0f32; let mut prev_z = 0.0f32;

    for i in 0..n_frames {
        let start = i * frame_size;
        if start + frame_size > body.len() { break; }
        let frame: ReplayFrame = *bytemuck::from_bytes(&body[start..start + frame_size]);
        if frame.ser as usize != ser_target { continue; }

        let dx = frame.x - prev_x; let dz = frame.z - prev_z;
        let d_step = (dx * dx + dz * dz).sqrt();

        println!("{:>6}  {:>8.1}  {:>8.1}  {:>7.2}  {:>7.3}  {:>7.3}  {:>6.3}  {:>6.1}  {:>6.1}  {:>7.1}  {:>7.3}  {:>7.3}",
            frame.tick, frame.x, frame.z, frame.speed,
            frame.throttle, frame.steer, frame.brake,
            frame.dist_l, frame.dist_r, frame.dist_f,
            frame.intent_0, frame.intent_1);

        let mut notas: Vec<&str> = Vec::new();
        if frame.dist_f < 5.0          { notas.push("PAREDE_PERTO"); }
        if frame.speed < 0.5           { notas.push("PARADO"); }
        if frame.intent_0.abs() > SAT_THRESH { notas.push("THR_SAT"); }
        if frame.intent_1.abs() > SAT_THRESH { notas.push("STR_SAT"); }
        if frame.brake > 0.5           { notas.push("FREANDO"); }
        if d_step < 0.1 && i > 0      { notas.push("NAO_MOVEU"); }
        if !notas.is_empty() {
            println!("         -> {}", notas.join(" | "));
        }

        prev_x = frame.x; prev_z = frame.z;
    }
    println!();
    println!("=== FIM DO REPLAY ===");
}

// ─── Ponto de entrada ─────────────────────────────────────────────────────────

fn main() {
    // Força ICD Radeon antes de qualquer wgpu::Instance (múltiplos ICDs → SIGSEGV).
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe { std::env::set_var("VK_ICD_FILENAMES", icd_radeon); }
    }

    // ── Args ──────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut n_seres:     usize         = 60;
    let mut n_ticks:     usize         = 3_000; // 3000 ticks ≈ 47 s de simulação @ 64Hz
    let mut replay_path: Option<String> = None;
    let mut replay_ser:  usize         = 0;
    {
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--seres"  => { i += 1; if i < args.len() { n_seres = args[i].parse().unwrap_or(60); } }
                "--ticks"  => { i += 1; if i < args.len() { n_ticks = args[i].parse().unwrap_or(3_000); } }
                "--replay" => { i += 1; if i < args.len() { replay_path = Some(args[i].clone()); } }
                "--ser"    => { i += 1; if i < args.len() { replay_ser  = args[i].parse().unwrap_or(0); } }
                _ => {}
            }
            i += 1;
        }
    }

    // ── Modo replay: carrega e exibe arquivo existente, sem rodar simulação ──
    if let Some(ref rpath) = replay_path {
        load_and_print_replay(rpath, replay_ser);
        return;
    }

    // ── Buffer de saída — tudo vai para stdout E para o arquivo summary ────────
    // Usamos um Vec<String> acumulado e no final gravamos tudo de uma vez.
    let mut report: Vec<String> = Vec::new();
    // rln! — imprime no stdout E acumula no Vec para salvar em summary.txt.
    // A versão sem argumento grava uma linha em branco.
    macro_rules! rln {
        () => {{
            println!();
            report.push(String::new());
        }};
        ($fmt:literal $(, $arg:expr)*) => {{
            let s = format!($fmt $(, $arg)*);
            println!("{}", s);
            report.push(s);
        }};
        ($s:expr) => {{
            let owned = $s.to_string();
            println!("{}", owned);
            report.push(owned);
        }};
    }

    rln!("=== GPU_REPORT: OBSERVATÓRIO DE CORRIDA ===");
    rln!("data: 2026-08-03");
    rln!("seres: {}   ticks: {}   dt: {:.6} s   amostragem: a cada {} ticks",
        n_seres, n_ticks, 1.0 / 64.0f32, SAMPLE_EVERY);

    // ── Diretório de saída ─────────────────────────────────────────────────────
    let out_dir = "telemetry/gpu_run";
    fs::create_dir_all(out_dir).expect("criar telemetry/gpu_run");

    // ── Mundo ─────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    // seed=9001 + largura=18m: mesmos valores do gpu_validate e gpu_loop canônico.
    let world    = VoxelWorld::new(9001, 18.0);
    let lap_dist = world.lap_distance;

    rln!();
    rln!("=== MAPA DA PISTA ===");
    rln!("seed: 9001   largura: 18 m   grade: {}×{} células de {:.1} m",
        world.size, world.size, world.cell_size);
    rln!("volta: {:.1} m   waypoints: {}",
        lap_dist, world.waypoints_world.len());

    let n_grass = world.grid.iter().filter(|&&v| v as u8 == 0).count();
    let n_road  = world.grid.iter().filter(|&&v| v as u8 == 1).count();
    let n_wall  = world.grid.iter().filter(|&&v| v as u8 == 2).count();
    let n_hole  = world.grid.iter().filter(|&&v| v as u8 == 3).count();
    rln!("voxels: road={}  wall={}  grass={}  hole={}  (total={})",
        n_road, n_wall, n_grass, n_hole, world.size * world.size);
    rln!("  [{:.0} ms para gerar o mundo]", t0.elapsed().as_secs_f64() * 1000.0);

    // ── DNA + engines ─────────────────────────────────────────────────────────
    let dna_opt = load_champion_dna("CHAMPION.dna");
    let engines: Vec<SovereignEngine> = (0..n_seres).map(|_| {
        let mut eng = SovereignEngine::new();
        if let Some(ref dna) = dna_opt {
            let w: Vec<f32> = if let Some(ref cr) = dna.cortex_raw {
                cr.iter().map(|&b| b as f32 / 63.5).collect()
            } else { dna.fast_weights.iter().map(|&b| b as f32 / 63.5).collect() };
            let n_l1 = eng.instinct_layer.weights.len().min(w.len());
            eng.instinct_layer.weights[..n_l1].copy_from_slice(&w[..n_l1]);
            if let Some(ref l2) = dna.cortex_l2_raw {
                let w2: Vec<f32> = l2.iter().map(|&b| b as f32 / 63.5).collect();
                let n_l2 = eng.cortex_l2.weights.len().min(w2.len());
                eng.cortex_l2.weights[..n_l2].copy_from_slice(&w2[..n_l2]);
            }
        }
        eng
    }).collect();
    use universe::kan::KAN_OUT_KNOTS;
    let kan_knots: [[f32; KAN_OUT_KNOTS]; 3] = dna_opt.as_ref()
        .map(|d| d.kan_knots)
        .unwrap_or_else(universe::evolution::dna_genesis::default_kan_knots);

    rln!();
    rln!("DNA: {}", if dna_opt.is_some() {
        "CHAMPION.dna carregado — pesos REAIS"
    } else {
        "SEM CHAMPION.dna — pesos ZEROS (seres nao se moverão por decisão própria)"
    });

    // ── GpuLoop ───────────────────────────────────────────────────────────────
    let t_gpu = Instant::now();
    let mut ctx = match GpuLoop::new(&world, &engines, n_seres, kan_knots) {
        Some(c) => c,
        None => { eprintln!("ERRO: sem GPU utilizavel."); std::process::exit(1); }
    };
    rln!("GPU: {}  [{:.0} ms init]", ctx.adapter_name, t_gpu.elapsed().as_secs_f64() * 1000.0);

    // ── Estado inicial ─────────────────────────────────────────────────────────
    let n_wp = world.waypoints_world.len();
    let mut states: Vec<BeingState> = (0..n_seres).map(|i| {
        let wp      = world.waypoints_world[(i * n_wp / n_seres).min(n_wp - 1)];
        let wp_next = world.waypoints_world[((i * n_wp / n_seres) + 1).min(n_wp - 1)];
        let dx = wp_next.x - wp.x; let dz = wp_next.y - wp.y;
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
        let fwd = [dx / len, 0.0, dz / len];
        let pos = [wp.x, 0.0, wp.y];
        BeingState {
            phys: make_physics_state(pos, fwd),
            prev_steer: 0.0, stagnation_ticks: 0,
            escape_ticks: 0, reverse_timer: 0.0, post_reverse_lockout: 0.0,
        }
    }).collect();

    let pos_inicial: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

    // ── Acumuladores ──────────────────────────────────────────────────────────
    let dt        = 1.0_f32 / 64.0;
    let mut dist_acum:  Vec<f64>  = vec![0.0; n_seres];
    let mut pos_prev:   Vec<[f32; 3]> = pos_inicial.clone();
    let mut speed_max:  Vec<f32>  = vec![0.0; n_seres];
    let mut speed_sum:  Vec<f64>  = vec![0.0; n_seres];
    let mut ticks_vivo: Vec<usize> = vec![0; n_seres];
    // Posição onde cada ser ficou travado pela última vez (speed < 0.5 m/s por > 100 ticks).
    let mut stuck_pos: Vec<Option<[f32; 3]>> = vec![None; n_seres];
    let mut stuck_acum: Vec<usize> = vec![0; n_seres];

    // Trajetórias: Vec por ser, cada entrada = (tick, x, z, speed, throttle, steer, brake).
    let mut traj: Vec<Vec<(usize, f32, f32, f32, f32, f32, f32)>> = vec![vec![]; n_seres];

    // Série temporal de velocidade dos primeiros PERFIL_N_SERES seres (para gráfico).
    // Armazenada como Vec<(tick, speed_m_s)>.
    let mut vel_series: Vec<Vec<(usize, f32)>> = vec![vec![]; PERFIL_N_SERES.min(n_seres)];

    // Diagnóstico de aceleração: ser 0, primeiros 200 ticks.
    let mut accel_diag: Vec<(usize, f32, f32, f32, f32)> = Vec::new();
    let mut speed_prev0 = 0.0f32;

    // ── Acumuladores novos: mapa de calor, replay, neural ────────────────────

    // Mapa de calor: todas as posições (x,z) de todos os seres em ticks amostrados.
    let mut heatmap_frames: Vec<(f32, f32)> = Vec::new();

    // Replay: frames amostrados a cada REPLAY_SAMPLE_EVERY ticks.
    let mut replay_frames: Vec<ReplayFrame> = Vec::new();

    // Distâncias laterais/frontais por ser por tick (injetadas na percepção).
    // Guardamos apenas os últimos STUCK_WINDOW ticks antes de cada travamento.
    // dist_window[i] = ring buffer de (tick, dist_l, dist_r, dist_f).
    const DIST_WINDOW: usize = 10;
    let mut dist_window: Vec<std::collections::VecDeque<(usize, f32, f32, f32)>> =
        (0..n_seres).map(|_| std::collections::VecDeque::with_capacity(DIST_WINDOW + 1)).collect();

    // Intent no travamento: capturado quando stuck_acum[i] == 100.
    // Guardamos o intent e as distâncias do momento do travamento.
    let mut stuck_intent: Vec<Option<Vec<f32>>> = vec![None; n_seres];
    let mut stuck_dist:   Vec<Option<[f32; 3]>> = vec![None; n_seres];

    // Rastreador de saturação persistente: alerta quando canal saturado por >=SAT_MIN_TICKS.
    // SENSORY_WAVE_SIZE = 223 (dimensões de intent).
    use universe::config::iara::brain::L1_NEURONS;
    let mut sat_tracker = SatTracker::new(n_seres, L1_NEURONS);

    // Último intent lido por ser (para snapshot de travamento).
    let mut last_intent: Vec<Vec<f32>> = vec![vec![0.0f32; L1_NEURONS]; n_seres];

    // Ultimo tick em que capturamos distâncias (para replay).
    let mut last_dist: Vec<[f32; 3]> = vec![[15.0, 15.0, 50.0]; n_seres];

    // ── Aquecimento ───────────────────────────────────────────────────────────
    rln!();
    rln!("Aquecendo (1 tick JIT)...");
    ctx.tick(&mut states, &engines, n_seres, dt, 0);
    // Reseta após aquecimento
    for i in 0..n_seres { pos_prev[i] = pos_of(&states[i].phys); }

    // ── Laço principal ────────────────────────────────────────────────────────
    rln!("Rodando {} ticks ({:.1} s simulados @ 64 Hz)...", n_ticks, n_ticks as f32 / 64.0);
    let t_loop = Instant::now();

    for tick_i in 0..n_ticks {
        let (ok, pedals) = ctx.tick(&mut states, &engines, n_seres, dt, tick_i + 1);
        if !ok { eprintln!("ERRO no tick {}", tick_i); break; }

        // Lê intent da GPU a cada tick para análise neural.
        // read_intent_after_submit é barato: o buffer já está pronto (poll(Wait) no tick).
        let intents_now = ctx.brain.read_intent_after_submit(n_seres);

        // Recalcula distâncias para este tick (reutiliza a lógica do tick).
        // Apenas salva no window — cálculo já foi feito dentro do tick().
        // Reutilizamos a posição atual + grade para recomputar.
        {
            let gs  = ctx.grid_size_cpu as usize;
            let cs  = ctx.cell_size_cpu;
            // Amostras laterais expandidas de 27m → 40m (conta: 40/28 m/s = 1.43 s antecipação).
            let lat_samp = [1.0f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 33.0, 40.0];
            let fwd_samp = [3.0f32, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0];
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    ctx.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };
            for i in 0..n_seres {
                let p = &states[i].phys;
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let perp_x = -fz; let perp_z = fx;
                let mut dist_l = 40.0f32;
                for &sd in &lat_samp {
                    if voxel_at(px - perp_x * sd, pz - perp_z * sd) == 2u32 { dist_l = sd; break; }
                }
                let mut dist_r = 40.0f32;
                for &sd in &lat_samp {
                    if voxel_at(px + perp_x * sd, pz + perp_z * sd) == 2u32 { dist_r = sd; break; }
                }
                let mut dist_f = 50.0f32;
                for &sd in &fwd_samp {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2u32 || v == 3u32 { dist_f = sd; break; }
                }
                last_dist[i] = [dist_l, dist_r, dist_f];
                // Janela deslizante de distâncias (últimos DIST_WINDOW ticks).
                if dist_window[i].len() >= DIST_WINDOW {
                    dist_window[i].pop_front();
                }
                dist_window[i].push_back((tick_i + 1, dist_l, dist_r, dist_f));
            }
        }

        // Atualiza last_intent e sat_tracker com o intent deste tick.
        for i in 0..n_seres {
            if i < intents_now.len() {
                last_intent[i].copy_from_slice(&intents_now[i][..L1_NEURONS.min(intents_now[i].len())]);
                sat_tracker.update(i, tick_i + 1, &intents_now[i][..L1_NEURONS.min(intents_now[i].len())]);
            }
        }

        // Acumula estatísticas por ser
        for i in 0..n_seres {
            let p1  = pos_of(&states[i].phys);
            let dx  = (p1[0] - pos_prev[i][0]) as f64;
            let dz  = (p1[2] - pos_prev[i][2]) as f64;
            let d   = (dx * dx + dz * dz).sqrt();
            dist_acum[i] += d;
            pos_prev[i]   = p1;

            let spd = (states[i].phys.vel[0].powi(2) + states[i].phys.vel[2].powi(2)).sqrt();
            if spd > speed_max[i] { speed_max[i] = spd; }
            speed_sum[i]  += spd as f64;
            ticks_vivo[i] += 1;

            // Detecção de travamento: > 100 ticks consecutivos com speed < 0.5 m/s.
            if spd < 0.5 {
                stuck_acum[i] += 1;
                if stuck_acum[i] == 100 {
                    // Marco o local onde ficou travado.
                    stuck_pos[i] = Some(p1);
                    // Captura intent e distâncias no momento do travamento.
                    if stuck_intent[i].is_none() {
                        stuck_intent[i] = Some(last_intent[i].clone());
                        stuck_dist[i]   = Some(last_dist[i]);
                    }
                }
            } else {
                stuck_acum[i] = 0;
            }
        }

        // Amostragem de trajetória a cada SAMPLE_EVERY ticks
        if (tick_i + 1) % SAMPLE_EVERY == 0 {
            for i in 0..n_seres {
                let p   = &states[i].phys;
                let spd = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let ped = pedals.get(i).cloned().unwrap_or([0.0; 3]);
                traj[i].push((tick_i + 1, p.pos[0], p.pos[2], spd, ped[0], ped[1], ped[2]));
            }
        }

        // Mapa de calor: posições de todos os seres a cada SAMPLE_EVERY ticks.
        if (tick_i + 1) % SAMPLE_EVERY == 0 {
            for i in 0..n_seres {
                let p = &states[i].phys;
                heatmap_frames.push((p.pos[0], p.pos[2]));
            }
        }

        // Replay: frames a cada REPLAY_SAMPLE_EVERY ticks (= SAMPLE_EVERY).
        if (tick_i + 1) % REPLAY_SAMPLE_EVERY == 0 {
            for i in 0..n_seres {
                let p   = &states[i].phys;
                let spd = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let ped = pedals.get(i).cloned().unwrap_or([0.0; 3]);
                let d   = last_dist[i];
                let it  = if i < intents_now.len() { &intents_now[i] } else { &[0.0f32; 96] };
                replay_frames.push(ReplayFrame {
                    tick:     (tick_i + 1) as u32,
                    ser:      i as u32,
                    x:        p.pos[0],
                    z:        p.pos[2],
                    speed:    spd,
                    throttle: ped[0],
                    steer:    ped[1],
                    brake:    ped[2],
                    dist_l:   d[0],
                    dist_r:   d[1],
                    dist_f:   d[2],
                    fwd_x:    p.fwd[0],
                    fwd_z:    p.fwd[2],
                    intent_0: it.get(0).cloned().unwrap_or(0.0),
                    intent_1: it.get(1).cloned().unwrap_or(0.0),
                    intent_2: it.get(2).cloned().unwrap_or(0.0),
                    _pad:     0.0,
                });
            }
        }

        // Série de velocidade para os primeiros PERFIL_N_SERES seres (toda tick)
        for i in 0..vel_series.len() {
            let spd = (states[i].phys.vel[0].powi(2) + states[i].phys.vel[2].powi(2)).sqrt();
            vel_series[i].push((tick_i + 1, spd));
        }

        // Diagnóstico de aceleração (ser 0, primeiros 200 ticks)
        if tick_i < 200 {
            let spd0   = (states[0].phys.vel[0].powi(2) + states[0].phys.vel[2].powi(2)).sqrt();
            let accel  = (spd0 - speed_prev0) / dt;
            let ped    = pedals.get(0).cloned().unwrap_or([0.0; 3]);
            accel_diag.push((tick_i + 1, spd0, accel, ped[0], ped[2]));
            speed_prev0 = spd0;
        }

        // Checkpoint a cada 1000 ticks
        if (tick_i + 1) % 1000 == 0 {
            let max_d = dist_acum.iter().cloned().fold(0.0f64, f64::max);
            rln!("  tick {:5}: dist_max={:.0} m ({:.2} voltas)",
                tick_i + 1, max_d, max_d / lap_dist as f64);
        }
    }

    let elapsed_loop = t_loop.elapsed().as_secs_f64();
    rln!("Laço concluído: {:.1} ms total, {:.3} ms/tick",
        elapsed_loop * 1000.0, elapsed_loop * 1000.0 / n_ticks as f64);

    // ── Posições finais ────────────────────────────────────────────────────────
    let pos_final: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

    // ── Ranking por distância ──────────────────────────────────────────────────
    let mut ordem: Vec<usize> = (0..n_seres).collect();
    ordem.sort_by(|&a, &b| dist_acum[b].partial_cmp(&dist_acum[a]).unwrap());

    let dist_max = dist_acum[ordem[0]];
    let dist_med = {
        let mut v = dist_acum.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let n_volltas  = dist_acum.iter().filter(|&&d| d >= lap_dist as f64).count();
    let n_moveu    = dist_acum.iter().filter(|&&d| d > 1.0).count();
    let n_travados = stuck_pos.iter().filter(|p| p.is_some()).count();

    // ── Verificações de sanidade ───────────────────────────────────────────────
    let mut alertas: Vec<String> = Vec::new();

    // Todos com a mesma distância?
    if let Some(msg) = check_homogeneidade(&dist_acum, "distancia") {
        alertas.push(msg);
    }
    // Todos com a mesma velocidade máxima?
    let vmax_f64: Vec<f64> = speed_max.iter().map(|&v| v as f64).collect();
    if let Some(msg) = check_homogeneidade(&vmax_f64, "speed_max") {
        alertas.push(msg);
    }
    // Nenhum se moveu?
    if n_moveu == 0 {
        alertas.push(format!(
            "⚠ CRÍTICO: NENHUM ser percorreu > 1 m — laço quebrado ou pesos nulos."
        ));
    } else if n_moveu < n_seres / 2 {
        alertas.push(format!(
            "⚠ Apenas {}/{} seres se moveram > 1 m.", n_moveu, n_seres
        ));
    }

    // ── Mapa ASCII ────────────────────────────────────────────────────────────
    // Mostra posições iniciais e finais dos top-N seres + posições de travamento.
    let n_mark     = n_seres.min(15);
    let top_ini: Vec<[f32; 3]>  = ordem.iter().take(n_mark).map(|&i| pos_inicial[i]).collect();
    let top_fin: Vec<[f32; 3]>  = ordem.iter().take(n_mark).map(|&i| pos_final[i]).collect();
    let stuck_vec: Vec<[f32; 3]> = stuck_pos.iter()
        .filter_map(|p| *p)
        .collect();

    let ascii_lines = render_ascii_track(&world, &top_ini, &top_fin, &stuck_vec, ASCII_COLS);

    rln!();
    rln!("=== MAPA ASCII DA PISTA (seed=9001, escala {}:1) ===", world.size / ASCII_COLS);
    rln!("Legenda: '#'=wall  '.'=road  ' '=grass  'S'=pos_inicial  'F'=pos_final  'T'=travado");
    rln!("(top {} seres marcados; 'T' = travado por >100 ticks = {:.1} s)", n_mark, 100.0 / 64.0);
    rln!();
    for line in &ascii_lines { rln!("{}", line); }
    rln!();

    // ── MAPA DE CALOR ─────────────────────────────────────────────────────────
    // Densidade de seres por célula ao longo de toda a corrida (posições amostradas).
    rln!("=== MAPA DE CALOR (densidade de seres por celula, {} amostras, 'X'=travamento) ===", heatmap_frames.len());
    rln!("Onde os seres passaram MAIS tempo — identifica gargalos de pista.");
    {
        let hg = build_heatmap(&world, &heatmap_frames, HEATMAP_COLS);
        let hm_lines = render_heatmap(&world, &hg, HEATMAP_COLS, &stuck_vec);
        for line in &hm_lines { rln!("{}", line); }
        // Sumário do mapa de calor: top-5 células mais quentes.
        let mut cell_heat: Vec<((usize, usize), usize)> = hg.iter().enumerate()
            .flat_map(|(r, row)| row.iter().enumerate().map(move |(c, &v)| ((r, c), v)))
            .filter(|&(_, v)| v > 0)
            .collect();
        cell_heat.sort_by(|a, b| b.1.cmp(&a.1));
        rln!();
        rln!("  Top-5 celulas mais quentes (linha,col display → hits):");
        for &((r, c), hits) in cell_heat.iter().take(5) {
            // Converter para coordenadas mundo
            let gs   = world.size;
            let step = (gs / HEATMAP_COLS).max(1);
            let wx = (c * step) as f32 * world.cell_size;
            let wz = (r * step) as f32 * world.cell_size;
            rln!("    ({:3},{:3}) x≈{:.0}m z≈{:.0}m  hits={}", r, c, wx, wz, hits);
        }
    }
    rln!();

    // ── ESTADO NEURAL NO TRAVAMENTO ───────────────────────────────────────────
    rln!("=== NEURÔNIOS NO TRAVAMENTO (intent no tick t=100 de estagnacao) ===");
    rln!("  Pergunta: cerebro TRAVADO (ponto fixo) ou REAGINDO mas preso?");
    {
        let snapshots: Vec<StuckBrainSnapshot> = (0..n_seres)
            .filter_map(|i| {
                let intent = stuck_intent[i].clone()?;
                let dist   = stuck_dist[i]?;
                Some(StuckBrainSnapshot {
                    ser:    i,
                    tick:   0, // tick exato varia; marcado como "no travamento"
                    intent,
                    dist_l: dist[0],
                    dist_r: dist[1],
                    dist_f: dist[2],
                })
            })
            .collect();

        let brain_lines = analyze_stuck_brain(&snapshots, L1_NEURONS);
        for line in brain_lines { rln!("{}", line); }
    }
    rln!();

    // ── ALERTAS DE SATURAÇÃO PERSISTENTE ─────────────────────────────────────
    rln!("=== ALERTAS DE SATURACAO PERSISTENTE (|v|>{:.2} por >={} ticks) ===",
        SAT_THRESH, SAT_MIN_TICKS);
    rln!("  Saturacao persistente = canal travado no mesmo sentido por muitos ticks = reflexo fixo.");
    if sat_tracker.alertas.is_empty() {
        rln!("  Nenhuma saturacao persistente detectada.");
    } else {
        rln!("  {} alertas (primeiros 20 mostrados):", sat_tracker.alertas.len());
        for a in sat_tracker.alertas.iter().take(20) {
            rln!("{}", a);
        }
        if sat_tracker.alertas.len() > 20 {
            rln!("  ... e mais {} alertas omitidos.", sat_tracker.alertas.len() - 20);
        }
    }
    rln!();

    // ── AUTO-DIAGNÓSTICO ─────────────────────────────────────────────────────
    rln!("=== AUTO-DIAGNOSTICO ===");
    rln!("  Checagem automatica de padroes suspeitos:");
    {
        let diags = auto_diagnostico(
            &stuck_pos,
            n_seres,
            &world,
            &stuck_intent,
            &stuck_dist,
            HEATMAP_COLS,
        );
        for d in diags { rln!("{}", d); }
    }
    rln!();

    // ── Alertas de sanidade ────────────────────────────────────────────────────
    if !alertas.is_empty() {
        rln!("=== ALERTAS DE SANIDADE ===");
        for a in &alertas { rln!("{}", a); }
        rln!();
    }

    // ── Resumo geral ──────────────────────────────────────────────────────────
    rln!("=== RESUMO GERAL ===");
    rln!("  ticks: {}   dt: {:.6} s   tempo_sim: {:.1} s",
        n_ticks, dt, n_ticks as f32 * dt);
    rln!("  dist_max: {:.1} m  ({:.2} voltas)", dist_max, dist_max / lap_dist as f64);
    rln!("  dist_med: {:.1} m  ({:.2} voltas)", dist_med, dist_med / lap_dist as f64);
    rln!("  seres com >= 1 volta completa: {}/{}", n_volltas, n_seres);
    rln!("  seres que se moveram (> 1 m): {}/{}", n_moveu, n_seres);
    rln!("  seres travados (>100 ticks parado): {}/{}", n_travados, n_seres);
    rln!();

    // ── Tabela por ser ────────────────────────────────────────────────────────
    rln!("=== TABELA POR SER (ordenado por distância) ===");
    rln!("{:>4} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>7} {:>7} {:>8} {:>8}",
        "ser", "dist(m)", "voltas", "ini_x", "ini_z", "fin_x", "fin_z",
        "vmax", "vmed", "arc(m)", "status");
    for &i in &ordem {
        let d    = dist_acum[i];
        let vmax = speed_max[i];
        let vmed = if ticks_vivo[i] > 0 { speed_sum[i] / ticks_vivo[i] as f64 } else { 0.0 };
        let arc  = arc_progress(&world.racing_line, pos_final[i]);
        let arc_str = if arc < 0.0 { "NÃO MEDIDO".to_string() } else { format!("{:.1}", arc) };
        let status = if d < 1.0 {
            "PARADO"
        } else if stuck_pos[i].is_some() {
            "TRAVOU"
        } else {
            "OK"
        };
        rln!("{:>4} {:>8.1} {:>6.2} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>7.1} {:>7.2} {:>8} {:>8}",
            i, d, d / lap_dist as f64,
            pos_inicial[i][0], pos_inicial[i][2],
            pos_final[i][0], pos_final[i][2],
            vmax, vmed, arc_str, status);
    }
    rln!();

    // ── Perfis de velocidade ASCII (3 seres) ──────────────────────────────────
    rln!("=== PERFIS DE VELOCIDADE (seres 0, 1, 2 — tick × m/s) ===");
    rln!("(série amostrada a cada tick; comprimida para {} colunas ASCII)", PERFIL_COLS);
    rln!("Serve para ver se o ser acelera, aonde perde velocidade e padrão de frenagem.");
    rln!();
    for (idx, ser_idx) in (0..vel_series.len()).enumerate() {
        let titulo = format!("ser[{}]  dist={:.0}m  vmax={:.1}m/s",
            ser_idx, dist_acum[ser_idx], speed_max[ser_idx]);
        let chart  = ascii_velocity_chart(&vel_series[idx], PERFIL_ROWS, PERFIL_COLS, &titulo);
        for line in chart { rln!("{}", line); }
        rln!();
    }

    // ── Histogramas ───────────────────────────────────────────────────────────
    rln!("=== HISTOGRAMA DE DISTÂNCIAS (m) ===");
    let hist_lines = ascii_histogram(&dist_acum, HIST_BINS, "distancia percorrida");
    for line in hist_lines { rln!("{}", line); }
    rln!();

    rln!("=== HISTOGRAMA DE VELOCIDADE MÁXIMA (m/s) ===");
    let hist_vmax = ascii_histogram(&vmax_f64, HIST_BINS, "vmax por ser");
    for line in hist_vmax { rln!("{}", line); }
    rln!();

    // Histograma de velocidade média
    let vmed_f64: Vec<f64> = (0..n_seres).map(|i| {
        if ticks_vivo[i] > 0 { speed_sum[i] / ticks_vivo[i] as f64 } else { 0.0 }
    }).collect();
    rln!("=== HISTOGRAMA DE VELOCIDADE MÉDIA (m/s) ===");
    let hist_vmed = ascii_histogram(&vmed_f64, HIST_BINS, "vmed por ser");
    for line in hist_vmed { rln!("{}", line); }
    rln!();

    // ── Grava trajectories.csv ─────────────────────────────────────────────────
    let traj_path = format!("{}/trajectories.csv", out_dir);
    {
        let mut f = fs::File::create(&traj_path).expect("trajectories.csv");
        // Colunas: tick, ser, x, z, speed_m_s, throttle, steer, brake
        writeln!(f, "tick,ser,x,z,speed_m_s,throttle,steer,brake").unwrap();
        for i in 0..n_seres {
            // Registra posição inicial (tick=0)
            writeln!(f, "0,{},{:.2},{:.2},0.000,0.000,0.000,0.000",
                i, pos_inicial[i][0], pos_inicial[i][2]).unwrap();
            // Amostras do laço
            for &(tick, x, z, spd, thr, str_, brk) in &traj[i] {
                writeln!(f, "{},{},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3}",
                    tick, i, x, z, spd, thr, str_, brk).unwrap();
            }
        }
    }
    rln!("CSV gravado: {} ({} linhas)", traj_path,
        traj.iter().map(|t| t.len()).sum::<usize>() + n_seres);

    // ── Grava accel_diag.txt ───────────────────────────────────────────────────
    let accel_path = format!("{}/accel_diag.txt", out_dir);
    {
        let mut f = fs::File::create(&accel_path).expect("accel_diag.txt");
        writeln!(f, "=== DIAGNÓSTICO DE ACELERAÇÃO (ser 0, primeiros 200 ticks) ===").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "Parâmetros: mass=1350 kg  max_torque=620 Nm  wheel_radius=0.33 m").unwrap();
        writeln!(f, "  gear=3 (2a marcha)  mu_peak=1.6").unwrap();
        let normal_load  = 1350.0f64 * 9.81 / 4.0;
        let mu           = 1.6f64;
        let accel_max    = mu * normal_load * 2.0 / 1350.0;
        writeln!(f, "  F_grip_2rodas = {:.0} N  =>  a_max_teo = {:.2} m/s^2",
            mu * normal_load * 2.0, accel_max).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "{:>5} {:>7} {:>9} {:>9} {:>7} {:>7}  nota",
            "tick", "t(s)", "v(m/s)", "a(m/s2)", "thr", "brk").unwrap();
        for &(tick, spd, accel, thr, brk) in &accel_diag {
            let nota = if thr < 0.05 { "throttle_zero" }
                       else if brk > 0.3 { "freando" }
                       else if spd < 0.05 { "parado" }
                       else if accel > accel_max as f32 * 0.7 { "ACCEL_OK" }
                       else { "" };
            writeln!(f, "{:>5} {:>7.3} {:>9.3} {:>9.3} {:>7.3} {:>7.3}  {}",
                tick, tick as f32 * dt, spd, accel, thr, brk, nota).unwrap();
        }
        writeln!(f).unwrap();
        let vmax0 = accel_diag.iter().map(|&(_, s, _, _, _)| s).fold(0.0f32, f32::max);
        let amax0 = accel_diag.iter().filter(|&&(_, _, a, _, _)| a > 0.0)
            .map(|&(_, _, a, _, _)| a).fold(0.0f32, f32::max);
        writeln!(f, "vmax0 nos primeiros 200 ticks: {:.1} m/s = {:.0} km/h", vmax0, vmax0 * 3.6).unwrap();
        writeln!(f, "amax0 medida: {:.2} m/s^2  (teo: {:.2} m/s^2  = {:.0}%)",
            amax0, accel_max, amax0 as f64 / accel_max * 100.0).unwrap();
    }
    rln!("Diagnóstico de aceleração: {}", accel_path);

    // ── Grava replay binário ───────────────────────────────────────────────────
    // Nome inclui timestamp para não sobrescrever corridas anteriores.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let replay_path_out = format!("{}/replay_{}.bin", out_dir, ts);
    match save_replay(&replay_path_out, n_seres, n_ticks, &replay_frames) {
        Ok(_) => rln!("Replay gravado: {} ({} frames)", replay_path_out, replay_frames.len()),
        Err(e) => rln!("ERRO ao gravar replay: {}", e),
    }

    // ── Grava summary.txt (cópia completa do relatório) ───────────────────────
    let summary_path = format!("{}/summary.txt", out_dir);
    {
        let mut f = fs::File::create(&summary_path).expect("summary.txt");
        for line in &report { writeln!(f, "{}", line).unwrap(); }
    }
    println!("Relatório completo gravado: {}", summary_path);

    // ── Linha final de sanidade ────────────────────────────────────────────────
    println!();
    println!("=== FIM DO RELATÓRIO ===");
    println!("Telemetria em: {}/", out_dir);
    println!("  summary.txt        — cópia completa do relatório acima");
    println!("  trajectories.csv   — tick,ser,x,z,speed,throttle,steer,brake (a cada {} ticks)", SAMPLE_EVERY);
    println!("  accel_diag.txt     — aceleração medida vs teórica (ser 0, 200 ticks)");
    println!("  replay_<ts>.bin    — replay binário; ver com --replay <arquivo> --ser N");
    if !alertas.is_empty() {
        println!();
        println!("ALERTAS ATIVOS: {}", alertas.len());
        for a in &alertas { println!("  {}", a); }
    }
    if !sat_tracker.alertas.is_empty() {
        println!();
        println!("SATURACOES PERSISTENTES: {} (ver secao ALERTAS DE SATURACAO no relatório)", sat_tracker.alertas.len());
    }
}
