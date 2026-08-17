//! # RetinaDigestGpu — host Rust para `retina_digest.wgsl`
//!
//! Substitui o readback de 340 KB/tick (retina inteira) + laço Rust que produzia, por
//! ser, o estado do CfC de percepção (ch192-207), `min_wall_ahead` e `edge_row6/7` —
//! ver o cabeçalho de `retina_digest.wgsl` para a motivação completa e a equação.
//!
//! # Uso esperado (context.rs::tick())
//!
//! ```text
//! 1. retina.dispatch                              ← escreve buf_retina_out (tick t)
//! 2. percept.bind_and_dispatch(...)                ← clear_buffer(buf_output) + canais 0-191
//! 3. retina_digest.bind_and_dispatch(enc, ...)     ← DEPOIS do clear acima; escreve
//!                                                     extra_out (min_wall/edge6/edge7 +
//!                                                     ch192-207, P1.8: NÃO MAIS em
//!                                                     buf_output — ver achado no
//!                                                     cabeçalho de retina_digest.wgsl)
//! 4. brain.bind_and_dispatch(...)                  ← lê buf_output (canais 0-191)
//! 5. queue.submit + device.poll(Wait)
//! 6. retina_digest.read_after_submit(...)          ← EXTRA_OUT_STRIDE floats/ser, não 1764
//! ```
//!
//! # P1.8 — ch192-207 não aliasa mais no `buf_output` do próximo ser
//!
//! Até P1.8, o passo 3 escrevia os 16 valores do estado CfC em `buf_output` nos
//! índices 192-207 — só que `buf_output` tem EXATAMENTE 48 words (192 bytes) por
//! ser, sem folga, então esse índice caía no INÍCIO do bloco do PRÓXIMO ser
//! (corrompendo, via atomicOr, os canais 0-15 dele — FAR/NEAR pool, lidos de
//! verdade por `brain.wgsl`). Como `brain.wgsl` nunca lê índice ≥ 192 de qualquer
//! forma (`const WAVE: u32 = 192u`), os 16 valores nunca influenciavam a rede —
//! só corrompiam o ser vizinho. Agora vão para `extra_out[3..19]`, espaço próprio
//! por ser. Ver o cabeçalho de `retina_digest.wgsl` para a evidência completa.
//!
//! `buf_output_shared` é o MESMO `buf_output` de `PerceptionGpu` — este módulo não o
//! possui, só recebe a referência a cada dispatch (igual ao padrão de
//! `bind_and_dispatch` em `perception.rs`, que recebe `buf_retina_fisica` por fora).

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::math::cfc_perception::{CFC_HIDDEN, CFC_WEIGHTS_LEN, flat_weights};
use crate::perception::RetinaPerception;

use super::core::{storage_entry, uniform_entry};

/// Uniform do kernel — layout deve bater com `Params` em `retina_digest.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsGpu {
    n_beings: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Contexto de compute persistente para o kernel `retina_digest.wgsl`.
///
/// Criado uma vez por corrida. Compartilha o `device`/`queue` do resto do
/// `gpu_evolve` — nunca abre um segundo dispositivo (mesma regra de `PerceptionGpu`).
pub struct RetinaDigestGpu {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Pesos do CfC de percepção, achatados via `math::cfc_perception::flat_weights`.
    /// Buffer ÚNICO, compartilhado por todos os seres — os pesos são fixos e idênticos
    /// para toda a população (ver doc de `flat_weights`). Upado uma vez em `new()`.
    buf_cfc_weights: wgpu::Buffer,
    /// Estado recorrente do CfC (h), `capacity × CFC_HIDDEN` f32. RESIDENTE na VRAM
    /// entre ticks — nunca desce no laço quente. Zerado por `reset()` entre gerações.
    buf_cfc_state: wgpu::Buffer,
    /// Saída pequena por ser: `[min_wall_ahead, edge_row6, edge_row7]`.
    buf_extra_out: wgpu::Buffer,
    /// Buffer MAP_READ espelhando `buf_extra_out` — separado porque STORAGE|MAP_READ
    /// é inválido em wgpu (mesmo padrão de `PerceptionGpu::buf_readback`).
    buf_extra_readback: wgpu::Buffer,
    pub capacity: usize,
}

/// Número de f32 por ser no buffer `extra_out`:
/// `[min_wall_ahead, edge_row6, edge_row7, ch192..ch207(16, estado CfC squashed)]`.
///
/// P1.8: cresceu de 3 para 19 — os 16 últimos slots são o estado CfC de percepção
/// que antes aliasava (via `write_channel`) para o `buf_output` do PRÓXIMO ser (ver
/// achado no cabeçalho deste arquivo e de `retina_digest.wgsl`). `read_after_submit`
/// devolve o array completo; `context.rs` hoje só consome `[0]`/`[1]`/`[2]`
/// (min_wall_ahead/edge_row6/edge_row7) — `[3..19]` fica disponível pra quem
/// precisar (ex.: religar como sensor de verdade em `brain.wgsl`, fora do escopo
/// deste conserto).
pub const EXTRA_OUT_STRIDE: usize = 19;

impl RetinaDigestGpu {
    /// Cria o contexto e sobe os pesos do CfC uma única vez.
    ///
    /// `device`/`queue` são emprestados do dono do device (mesmo padrão de
    /// `PerceptionGpu::new`) — nunca cria um segundo dispositivo wgpu.
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, capacity: usize) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retina-digest"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("retina_digest.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retina-digest-layout"),
            entries: &[
                uniform_entry(0),        // params
                storage_entry(1, true),  // retina física (read)
                storage_entry(2, true),  // cfc_weights (read)
                storage_entry(3, false), // cfc_state (read_write, persistente)
                storage_entry(4, false), // output — MESMO buf_output da percepção (atomicOr)
                storage_entry(5, false), // extra_out (read_write)
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("retina-digest-pipeline"),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Pesos do CfC — buffer único, achatado, upado uma vez ──────────────────
        // RetinaPerception::new() usa seed FIXA (0xCAFE_BABE_DEAD_BEEF) — todos os
        // seres do gpu_evolve compartilham os mesmos pesos de CfC de percepção (ver
        // achado em math/cfc_perception.rs). Se isso mudar (perception_weights
        // plugado no Genome), este upload precisa virar per-being — o teste
        // `duas_instancias_tem_os_mesmos_pesos` em cfc_perception.rs quebra primeiro
        // e avisa quem estiver mexendo aqui.
        let weights = flat_weights(&RetinaPerception::new().pipeline.cfc);
        debug_assert_eq!(weights.len(), CFC_WEIGHTS_LEN);
        let buf_cfc_weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retina-digest-cfc-weights"),
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cap = capacity.max(1);
        let state_zeros: Vec<f32> = vec![0.0; cap * CFC_HIDDEN];
        let buf_cfc_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retina-digest-cfc-state"),
            contents: bytemuck::cast_slice(&state_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let extra_bytes = (cap * EXTRA_OUT_STRIDE * std::mem::size_of::<f32>()) as u64;
        let buf_extra_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("retina-digest-extra-out"),
            size: extra_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_extra_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("retina-digest-extra-readback"),
            size: extra_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            layout,
            buf_cfc_weights,
            buf_cfc_state,
            buf_extra_out,
            buf_extra_readback,
            capacity: cap,
        }
    }

    /// Grava o dispatch no `CommandEncoder` externo, SEM submit nem poll — mesmo
    /// contrato de `PerceptionGpu::bind_and_dispatch`.
    ///
    /// # Pré-condições (ordem no encoder)
    ///
    /// 1. `buf_retina_fisica` já foi escrito pelo dispatch da retina NESTE tick.
    /// 2. `buf_output_shared` já foi zerado (`enc.clear_buffer`) — isso acontece
    ///    dentro de `PerceptionGpu::bind_and_dispatch`, então este método deve ser
    ///    chamado DEPOIS dele no mesmo encoder. Escrever antes do clear perderia os
    ///    canais 192-207 (o clear apagaria o que este kernel gravou).
    ///
    /// Retorna os bytes de `extra_out` preenchidos (para o `copy_buffer_to_buffer`
    /// que o chamador grava a seguir, ou já grava aqui — ver corpo).
    pub fn bind_and_dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
        buf_retina_fisica: &wgpu::Buffer,
        buf_output_shared: &wgpu::Buffer,
    ) -> u64 {
        let n = n.min(self.capacity);
        if n == 0 {
            return 0;
        }

        let params = ParamsGpu {
            n_beings: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retina-digest-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retina-digest-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_retina_fisica.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_cfc_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_cfc_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_output_shared.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_extra_out.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-retina-digest"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // workgroup_size(1): 1 thread por ser — mesmo padrão de perception_temporal.wgsl.
            pass.dispatch_workgroups(n as u32, 1, 1);
        }

        let bytes_extra = (n * EXTRA_OUT_STRIDE * std::mem::size_of::<f32>()) as u64;
        enc.copy_buffer_to_buffer(
            &self.buf_extra_out,
            0,
            &self.buf_extra_readback,
            0,
            bytes_extra,
        );
        let _ = queue; // reservado para simetria com PerceptionGpu::bind_and_dispatch (uniform via write_buffer futuro)
        bytes_extra
    }

    /// Lê `extra_out` de volta — chamar DEPOIS de `queue.submit` + `device.poll(Wait)`
    /// do encoder que continha `bind_and_dispatch`. Mesmo padrão de
    /// `PerceptionGpu::read_after_submit` (map_async → poll(Wait) → recv — NUNCA na
    /// ordem inversa, é o deadlock documentado duas vezes nesta base).
    ///
    /// Retorna `[min_wall_ahead, edge_row6, edge_row7]` por ser.
    pub fn read_after_submit(
        &self,
        device: &wgpu::Device,
        n: usize,
        bytes: u64,
    ) -> Vec<[f32; EXTRA_OUT_STRIDE]> {
        let n = n.min(self.capacity);
        if n == 0 || bytes == 0 {
            return Vec::new();
        }

        let slice = self.buf_extra_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let mut out = vec![[0.0f32; EXTRA_OUT_STRIDE]; n];
        // P1.8: copia a linha INTEIRA (`EXTRA_OUT_STRIDE`=19 floats), não só os 3
        // primeiros — bug introduzido e pego na hora por
        // `retina_digest_bate_com_cpu_ao_longo_de_varios_ticks` (--ignored): com o
        // loop antigo hardcoded em [0],[1],[2], os slots [3..19) (ch192-207) ficavam
        // sempre no `0.0` de inicialização do `Vec`, nunca no valor real escrito
        // pela GPU. Registro aqui porque é exatamente a classe de erro (índice
        // hardcoded que não acompanha o stride) que este projeto já pagou caro antes.
        for i in 0..n {
            let base = i * EXTRA_OUT_STRIDE;
            out[i].copy_from_slice(&floats[base..base + EXTRA_OUT_STRIDE]);
        }

        drop(data);
        self.buf_extra_readback.unmap();
        out
    }

    /// Zera o estado recorrente do CfC para os primeiros `n` seres.
    ///
    /// Chamado entre gerações (mesmo ponto onde `context.rs::reset_cfc` antes recriava
    /// `RetinaPerception::new()` por ser). Usa `write_buffer` — sem encoder, mesmo
    /// padrão de `BrainGpu::reset_cfc`. CRÍTICO: esquecer isto reintroduz o bug
    /// documentado em 2026-08-11 (filho herdando estado CfC do pai, reavaliação do
    /// elite não-determinística).
    pub fn reset(&self, queue: &wgpu::Queue, n: usize) {
        let n = n.min(self.capacity);
        if n == 0 {
            return;
        }
        let zeros: Vec<u8> = vec![0u8; n * CFC_HIDDEN * std::mem::size_of::<f32>()];
        queue.write_buffer(&self.buf_cfc_state, 0, &zeros);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `EXTRA_OUT_STRIDE` é o contrato com `context.rs` — mudar sem atualizar os dois
    /// lados quebra em silêncio (lê lixo do próximo ser). Trava o valor aqui.
    /// P1.8: 3→19 (16 slots novos para o estado CfC que deixou de aliasar em
    /// `buf_output` — ver doc da constante).
    #[test]
    fn extra_out_stride_e_dezenove() {
        assert_eq!(EXTRA_OUT_STRIDE, 19);
    }
}

// ─── Parte 2: parity CPU vs GPU — marcado #[ignore] (ver módulo `retina_digest_parity`) ──
#[cfg(test)]
#[path = "retina_digest_parity_test.rs"]
mod retina_digest_parity_test;
