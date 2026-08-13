//! Simulação na GPU, sem parte gráfica.
//!
//! # A arquitetura que o Leonardo pediu
//!
//! > *"preciso que rode na gpu sem a parte grafica mesmo, pq era para ter isso na base do
//! > projeto, formas de visualizar com bevy e sem, e a sem não é para renderizar o bevy —
//! > bevy seria só um frontend que exibe em uma engine gráfica o que está acontecendo"*
//!
//! Ou seja: o núcleo é compute, e o Bevy é uma janela opcional sobre ele. Este módulo é o
//! começo desse núcleo — ele fala com a GPU por `wgpu` (já na árvore via Bevy) **sem criar
//! surface, sem swapchain e sem janela**, então roda com `env -u DISPLAY`, em cron, em
//! máquina sem sessão gráfica.
//!
//! # Por que a retina primeiro
//!
//! Medido em 2440 amostras de PERF das corridas de hoje:
//!
//! | componente | fração do tempo |
//! |---|---|
//! | **retina** | **57,0%** |
//! | física e resto | 26,9% |
//! | cérebro | 14,3% |
//! | percepção | 1,9% |
//!
//! A retina é 14 linhas × 126 colunas × N seres — com 60 seres, **105.840 marchas DDA por
//! tick**, todas independentes. É o trabalho mais paralelo do simulador e o mais caro. O
//! cérebro, que é o que a tese estuda, custa um quarto disso.
//!
//! # A regra inegociável
//!
//! O kernel tem de dar o MESMO resultado que a CPU, pixel por pixel. Um caminho GPU que
//! responde diferente em silêncio invalidaria todo experimento já medido — e depois de um dia
//! inteiro descobrindo constantes que carregavam suposições mortas, essa é exatamente a
//! classe de erro que não pode entrar. `testes::gpu_igual_a_cpu` trava isso.

// ── Núcleo de simulação em GPU, escrito em paralelo por agentes ───────────────
// Cada módulo traz kernel WGSL + host wgpu. A regra é a mesma em todos: equivalência
// com a CPU demonstrada antes de qualquer alegação de ganho.
pub mod physics;
pub mod brain;
pub mod perception;
pub mod core;
pub mod evolution;
pub mod cortex_sheet;
pub mod slow_brain;
pub mod retina_types;

use std::borrow::Cow;

use bevy::math::{Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::config::iara::vision::{RETINA_COLS, RETINA_ROWS, RETINA_TOTAL};
use crate::world::{VoxelType, VoxelWorld};

/// Recurso Bevy que segura o contexto de GPU vivo entre ticks.
///
/// Recriar o dispositivo e recompilar o shader por tick anularia todo o ganho, então isto
/// vive na aplicação. A assinatura do mundo permite detectar troca de pista e resubir a
/// grade — se isso passasse batido, os seres veriam a pista ANTERIOR e nada acusaria.
#[derive(bevy::prelude::Resource, Default)]
pub struct RetinaGpuRes {
    pub ctx: Option<RetinaGpu>,
    pub signature: u64,
}

impl RetinaGpuRes {
    /// Devolve um contexto válido para este mundo, criando ou recriando se preciso.
    /// `None` = sem GPU utilizável; o chamador cai na CPU em vez de derrubar a corrida.
    pub fn get_or_create(&mut self, world: &VoxelWorld, capacity: usize) -> Option<&RetinaGpu> {
        let sig = world_signature(world);
        let precisa = self.ctx.is_none()
            || self.signature != sig
            || self.ctx.as_ref().map(|c| c.capacity < capacity).unwrap_or(true);
        if precisa {
            self.ctx = RetinaGpu::new(world, capacity.max(64));
            self.signature = sig;
            if let Some(c) = &self.ctx {
                println!("🖥  retina na GPU: {} ({} seres)", c.adapter, capacity);
            } else {
                println!("⚠️  sem GPU utilizável — retina fica na CPU");
            }
        }
        self.ctx.as_ref()
    }
}

/// Barata e suficiente: tamanho, comprimento da volta e uma amostragem da grade.
/// Exposta como pub para que motor.rs possa passar a mesma assinatura ao contexto de
/// percepção GPU sem recomputar ou duplicar a lógica.
pub fn world_signature(world: &VoxelWorld) -> u64 {
    let mut h = world.size as u64 ^ ((world.lap_distance as u64) << 20);
    let step = (world.grid.len() / 512).max(1);
    for (i, v) in world.grid.iter().step_by(step).enumerate() {
        h = h.wrapping_mul(1099511628211).wrapping_add((*v as u64) ^ i as u64);
    }
    h
}

/// Um pixel da retina, em grandezas FÍSICAS: o que um lidar entregaria.
///
/// A retina antiga devolvia um `i8` produzido por uma curva de salência escrita à mão, que
/// media 4 níveis úteis para pista e 3 para grama (normalizava por 1200 m num raio travado
/// em 33,3 m) e punha parede no topo da escala, de modo que o `max`-pool do consumo apagava
/// a pista de qualquer bin com parede. Agora a decisão de o que importa sai daqui.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pixel {
    /// Distância até a primeira superfície opaca, em metros.
    pub dist_m: f32,
    /// 0 = Grass, 1 = Road, 2 = Wall, 3 = Hole (mesma ordem de `VoxelType`).
    pub material: u8,
}

/// A codificação ANTIGA, reconstruída a partir do físico.
///
/// Existe por dois motivos, os dois de método: provar que o kernel de GPU é equivalente à
/// CPU pixel a pixel, e permitir comparar física × codificada num A/B em vez de assumir que
/// a nova é melhor. Assumir sem medir foi exatamente o erro de quem escreveu o 1200 m.
pub fn legacy_encoding(px: Pixel) -> i8 {
    let d = px.dist_m;
    let depth_fade = (d / 1200.0).clamp(0.0, 1.0);
    let v = match px.material {
        2 => {
            if d < 30.0 { 127.0 - ((d / 30.0).clamp(0.0, 1.0).sqrt() * 72.0) }
            else if d < 120.0 { 55.0 - (((d - 30.0) / 90.0).clamp(0.0, 1.0) * 55.0) }
            else if d < 1200.0 { 0.0 - (((d - 120.0) / 480.0).clamp(0.0, 1.0) * 25.0) }
            else { -25.0 - (((d - 1200.0) / 1200.0).clamp(0.0, 1.0) * 25.0) }
        }
        0 => 40.0 - (depth_fade * 40.0),
        1 => -128.0 + (depth_fade * 96.0),
        _ => 90.0 - ((d / 50.0).clamp(0.0, 1.0) * 30.0),
    };
    v.clamp(-128.0, 127.0) as i8
}

/// Postura de um ser no instante do disparo: onde está e para onde olha.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub pos: Vec3,
    pub forward: Vec2,
    pub right: Vec2,
    pub gaze_yaw: f32,
}

// Deduplicado (E3): era a quinta cópia idêntica de `RetinaParamsGpu`/`PoseRaw`
// (as outras quatro estavam em gpu_evolve.rs, gpu_loop.rs, cortex_evolve.rs e —
// antes ainda — dentro deste mesmo arquivo sob nomes diferentes). Alias local para
// não reescrever todos os literais `ParamsGpu { .. }` abaixo.
use crate::gpu::retina_types::{RetinaParamsGpu as ParamsGpu, PoseRaw};

/// Contexto de compute persistente. Criar isto custa (abre dispositivo, compila shader), então
/// vive durante toda a corrida — recriar por tick anularia o ganho.
pub struct RetinaGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buf_grid: wgpu::Buffer,
    grid_size: u32,
    cell_size: f32,
    pub capacity: usize,
    buf_poses: wgpu::Buffer,
    buf_output: wgpu::Buffer,
    buf_readback: wgpu::Buffer,
    pub adapter: String,
}

impl RetinaGpu {
    /// Abre a GPU e sobe o mundo (estático por pista — sobe uma vez, não por tick).
    ///
    /// Devolve `None` quando não há GPU utilizável, para o chamador poder cair na CPU sem
    /// derrubar a simulação. Corrida de dias não pode morrer por falta de dispositivo.
    pub fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        // `compatible_surface: None` é o ponto: pedimos adaptador para COMPUTE, sem janela.
        let adapter_obj = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let nome = adapter_obj.get_info().name.clone();
        // percepcao.wgsl usa 6 storage buffers — acima do limite padrão de downlevel (4).
        let mut limites = wgpu::Limits::downlevel_defaults();
        limites.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter_obj.request_device(&wgpu::DeviceDescriptor {
            label: Some("iara-compute"),
            required_features: wgpu::Features::empty(),
            required_limits: limites,
            memory_hints: wgpu::MemoryHints::Performance,
        }, None))
        .ok()?;

        let modulo = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("retina.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("retina-pipeline"),
            layout: Some(&pipe_layout),
            module: &modulo,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bytes_poses = (capacity * std::mem::size_of::<PoseRaw>()) as u64;
        let bytes_output = (capacity * RETINA_TOTAL * 4) as u64;
        let buf_poses = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("poses"),
            size: bytes_poses,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: bytes_output,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes_output,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            layout,
            buf_grid,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            capacity,
            buf_poses,
            buf_output,
            buf_readback,
            adapter: nome,
        })
    }

    /// Uma retina por ser, num único despacho. Devolve FÍSICA: distância e material.
    pub fn render(&self, poses_in: &[Pose]) -> Vec<Vec<Pixel>> {
        let n = poses_in.len().min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        let mut pitches = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches[row / 4][row % 4] = crate::visao::pitch(row);
        }
        let params = ParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (crate::visao::fov_deg() / 2.0).to_radians(),
            pitches,
        };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pose_data: Vec<PoseRaw> = poses_in[..n]
            .iter()
            .map(|p| PoseRaw {
                pos_gaze: [p.pos.x, p.pos.y, p.pos.z, p.gaze_yaw],
                dirs: [p.forward.x, p.forward.y, p.right.x, p.right.y],
            })
            .collect();
        self.queue.write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&pose_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_poses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_output.as_entire_binding() },
            ],
        });

        let bytes = (n * RETINA_TOTAL * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 64 threads por grupo: 2 waves de 32 no RDNA2, que é o wave size desta GPU.
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = self.buf_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[u32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let mut r = Vec::with_capacity(RETINA_TOTAL);
            for p in 0..RETINA_TOTAL {
                let bruto = vals[s * RETINA_TOTAL + p];
                r.push(Pixel {
                    dist_m: (bruto >> 8) as f32 / 1000.0,
                    material: (bruto & 0xFF) as u8,
                });
            }
            out.push(r);
        }
        drop(data);
        self.buf_readback.unmap();
        out
    }

    /// O buffer de saída da retina, para outro kernel consumir DIRETO na VRAM.
    ///
    /// É o ponto do porte da percepção: sem isto, os 1764 pixels por ser voltariam para a
    /// CPU só para serem reduzidos a 223 canais e subirem de novo.
    pub fn output_buffer(&self) -> &wgpu::Buffer { &self.buf_output }

    /// O dispositivo e a fila, para os outros kernels COMPARTILHAREM em vez de abrir outro.
    /// Abrir um segundo VkDevice na mesma GPU desperdiça memória e alguns drivers recusam.
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

    /// A retina traduzida para a codificação antiga — só para a prova de equivalência e
    /// para o A/B contra a física. Não use em caminho novo.
    pub fn render_legacy(&self, poses_in: &[Pose]) -> Vec<Vec<i8>> {
        self.render(poses_in).into_iter()
            .map(|r| r.into_iter().map(legacy_encoding).collect())
            .collect()
    }

    /// Grava o dispatch da retina num `CommandEncoder` externo SEM fazer submit nem poll.
    ///
    /// # Por que existe
    ///
    /// O `render_legacy()` fazia encoder + submit + poll(Wait) próprios, adicionando uma
    /// barreira de sincronização. Com `dispatch_into`, a retina e a percepção podem ser
    /// encadeadas no mesmo encoder e submetidas uma só vez.
    ///
    /// O resultado fica em `self.buf_output` (VRAM) e em `self.buf_readback` (via copy no
    /// mesmo encoder), mas o map só pode acontecer DEPOIS do submit + poll do chamador.
    ///
    /// # Fluxo esperado
    ///
    /// ```text
    /// 1. Gravar poses via queue.write_buffer() ANTES de chamar este método
    /// 2. dispatch_into(&mut enc, n, device)     ← dispatch retina
    /// 3. percept.bind_and_dispatch(&mut enc, ...) ← dispatch percepção
    /// 4. queue.submit(enc.finish())              ← UM submit
    /// 5. device.poll(Wait)                       ← UM bloqueio
    /// 6. read_legacy_after_submit(n)             ← resultado retina
    /// 7. percept.read_after_submit(n, bytes_out) ← resultado percepção
    /// ```
    ///
    /// Retorna N seres despachados (para o chamador chamar `read_legacy_after_submit(N)`).
    pub fn dispatch_into(
        &self,
        enc:      &mut wgpu::CommandEncoder,
        poses_in: &[Pose],
    ) -> usize {
        let n = poses_in.len().min(self.capacity);
        if n == 0 { return 0; }

        // Prepara uniform + poses (write_buffer não bloqueia — os dados ficam na fila até submit).
        let mut pitches = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches[row / 4][row % 4] = crate::visao::pitch(row);
        }
        let params = ParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (crate::visao::fov_deg() / 2.0).to_radians(),
            pitches,
        };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retina-params-chain"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pose_data: Vec<PoseRaw> = poses_in[..n]
            .iter()
            .map(|p| PoseRaw {
                pos_gaze: [p.pos.x, p.pos.y, p.pos.z, p.gaze_yaw],
                dirs: [p.forward.x, p.forward.y, p.right.x, p.right.y],
            })
            .collect();
        // write_buffer: CPU→GPU staging, sem bloqueio até submit.
        self.queue.write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&pose_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_poses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_output.as_entire_binding() },
            ],
        });

        let bytes = (n * RETINA_TOTAL * 4) as u64;
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 64 threads por grupo: 2 waves de 32 no RDNA2 (wave size desta GPU).
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        // Copia para readback no mesmo encoder — DMA sem round-trip pela CPU.
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes);

        n
    }

    /// Lê buf_readback da retina e decodifica para legacy [i8; RETINA_TOTAL] × N.
    ///
    /// DEVE ser chamado após `device.poll(Wait)` do encoder que continha `dispatch_into`.
    /// Sem o poll, o buffer não está disponível para map e o método panickará.
    pub fn read_legacy_after_submit(&self, n: usize) -> Vec<[i8; RETINA_TOTAL]> {
        if n == 0 { return Vec::new(); }
        let bytes = (n * RETINA_TOTAL * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        // Poll já foi feito pelo chamador com Wait — o recv() deve completar imediatamente.
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[u32] = bytemuck::cast_slice(&data);
        let mut out = vec![[0i8; RETINA_TOTAL]; n];
        for s in 0..n {
            for p in 0..RETINA_TOTAL {
                let bruto = vals[s * RETINA_TOTAL + p];
                let px = Pixel {
                    dist_m: (bruto >> 8) as f32 / 1000.0,
                    material: (bruto & 0xFF) as u8,
                };
                out[s][p] = legacy_encoding(px);
            }
        }
        drop(data);
        self.buf_readback.unmap();
        out
    }
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

/// Só para o teste de equivalência não depender de detalhe interno.
pub const COLS: usize = RETINA_COLS;

#[allow(dead_code)]
fn _tipos_usados(_: VoxelType) {}
