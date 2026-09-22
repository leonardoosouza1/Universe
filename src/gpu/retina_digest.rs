//! # RetinaDigestGpu — Rust host for `retina_digest.wgsl`
//!
//! Replaces the 340 KB/tick readback (whole retina) + Rust loop that produced, per
//! being, the perception CfC state (ch192-207), `min_wall_ahead` and `edge_row6/7` —
//! see the header of `retina_digest.wgsl` for the full motivation and the equation.
//!
//! # Expected usage (context.rs::tick())
//!
//! ```text
//! 1. retina.dispatch                              ← writes buf_retina_out (tick t)
//! 2. percept.bind_and_dispatch(...)                ← clear_buffer(buf_output) + channels 0-191
//! 3. retina_digest.bind_and_dispatch(enc, ...)     ← AFTER the clear above; writes
//!                                                     extra_out (min_wall/edge6/edge7 +
//!                                                     ch192-207, P1.8: NO LONGER into
//!                                                     buf_output — see the finding in
//!                                                     the header of retina_digest.wgsl)
//! 4. brain.bind_and_dispatch(...)                  ← reads buf_output (channels 0-191)
//! 5. queue.submit + device.poll(Wait)
//! 6. retina_digest.read_after_submit(...)          ← EXTRA_OUT_STRIDE floats/being, not 1764
//! ```
//!
//! # P1.8 — ch192-207 no longer aliases into the next being's `buf_output`
//!
//! Until P1.8, step 3 wrote the 16 CfC state values into `buf_output` at
//! indices 192-207 — but `buf_output` has EXACTLY 48 words (192 bytes) per
//! being, with no slack, so that index landed at the START of the NEXT being's
//! block (corrupting, via atomicOr, its channels 0-15 — FAR/NEAR pool, actually
//! read by `brain.wgsl`). Since `brain.wgsl` never reads index ≥ 192 anyway
//! (`const WAVE: u32 = 192u`), the 16 values never influenced the network —
//! they only corrupted the neighboring being. They now go to `extra_out[3..19]`,
//! each being's own space. See the header of `retina_digest.wgsl` for the full evidence.
//!
//! `buf_output_shared` is the SAME `buf_output` as `PerceptionGpu`'s — this module does
//! not own it, it only receives the reference at each dispatch (same pattern as
//! `bind_and_dispatch` in `perception.rs`, which receives `buf_retina_fisica` from outside).

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::math::cfc_perception::{flat_weights, CFC_HIDDEN, CFC_WEIGHTS_LEN};
use crate::perception::RetinaPerception;

use super::core::{storage_entry, uniform_entry};

/// Kernel uniform — layout must match `Params` in `retina_digest.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsGpu {
    n_beings: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Persistent compute context for the `retina_digest.wgsl` kernel.
///
/// Created once per race. Shares the `device`/`queue` of the rest of
/// `gpu_evolve` — never opens a second device (same rule as `PerceptionGpu`).
pub struct RetinaDigestGpu {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Perception CfC weights, flattened via `math::cfc_perception::flat_weights`.
    /// SINGLE buffer, shared by all beings — the weights are fixed and identical
    /// for the whole population (see the `flat_weights` doc). Uploaded once in `new()`.
    buf_cfc_weights: wgpu::Buffer,
    /// Recurrent CfC state (h), `capacity × CFC_HIDDEN` f32. RESIDENT in VRAM
    /// between ticks — never comes down in the hot loop. Zeroed by `reset()` between generations.
    buf_cfc_state: wgpu::Buffer,
    /// Small per-being output: `[min_wall_ahead, edge_row6, edge_row7]`.
    buf_extra_out: wgpu::Buffer,
    /// MAP_READ buffer mirroring `buf_extra_out` — separate because STORAGE|MAP_READ
    /// is invalid in wgpu (same pattern as `PerceptionGpu::buf_readback`).
    buf_extra_readback: wgpu::Buffer,
    pub capacity: usize,
}

/// Number of f32 per being in the `extra_out` buffer:
/// `[min_wall_ahead, edge_row6, edge_row7, ch192..ch207(16, squashed CfC state)]`.
///
/// P1.8: grew from 3 to 19 — the last 16 slots are the perception CfC state
/// that previously aliased (via `write_channel`) into the NEXT being's `buf_output` (see
/// the finding in this file's header and in `retina_digest.wgsl`). `read_after_submit`
/// returns the full array; `context.rs` today only consumes `[0]`/`[1]`/`[2]`
/// (min_wall_ahead/edge_row6/edge_row7) — `[3..19]` remains available to whoever
/// needs it (e.g. re-wiring it as a true sensor in `brain.wgsl`, out of scope
/// for this fix).
pub const EXTRA_OUT_STRIDE: usize = 19;

impl RetinaDigestGpu {
    /// Previous dispatch's 19-float extras, for resident sensor adapters.
    pub fn extra_buffer(&self) -> &wgpu::Buffer {
        &self.buf_extra_out
    }

    /// Creates the context and uploads the CfC weights a single time.
    ///
    /// `device`/`queue` are borrowed from the device owner (same pattern as
    /// `PerceptionGpu::new`) — never creates a second wgpu device.
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, capacity: usize) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retina-digest"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("retina_digest.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retina-digest-layout"),
            entries: &[
                uniform_entry(0),        // params
                storage_entry(1, true),  // physical retina (read)
                storage_entry(2, true),  // cfc_weights (read)
                storage_entry(3, false), // cfc_state (read_write, persistent)
                storage_entry(4, false), // output — SAME buf_output as perception's (atomicOr)
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

        // ── CfC weights — single buffer, flattened, uploaded once ─────────────────
        // RetinaPerception::new() uses a FIXED seed (0xCAFE_BABE_DEAD_BEEF) — all
        // gpu_evolve beings share the same perception CfC weights (see the
        // finding in math/cfc_perception.rs). If that changes (perception_weights
        // plugged into the Genome), this upload must become per-being — the test
        // `duas_instancias_tem_os_mesmos_pesos` in cfc_perception.rs breaks first
        // and warns whoever is touching this.
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
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
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

    /// Records the dispatch into the external `CommandEncoder`, WITHOUT submit or poll —
    /// same contract as `PerceptionGpu::bind_and_dispatch`.
    ///
    /// # Preconditions (order in the encoder)
    ///
    /// 1. `buf_retina_fisica` was already written by the retina dispatch THIS tick.
    /// 2. `buf_output_shared` was already zeroed (`enc.clear_buffer`) — that happens
    ///    inside `PerceptionGpu::bind_and_dispatch`, so this method must be
    ///    called AFTER it in the same encoder. Writing before the clear would lose
    ///    channels 192-207 (the clear would erase what this kernel wrote).
    ///
    /// Returns the bytes of `extra_out` filled in (for the `copy_buffer_to_buffer`
    /// the caller records next, or records here already — see the body).
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
            // workgroup_size(1): 1 thread per being — same pattern as perception_temporal.wgsl.
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
        let _ = queue; // reserved for symmetry with PerceptionGpu::bind_and_dispatch (future uniform via write_buffer)
        bytes_extra
    }

    /// Starts mapping the compact retina digest staging buffer.
    pub fn request_read(
        &self,
        n: usize,
        bytes: u64,
    ) -> Option<crate::gpu::readback::ReadbackTicket> {
        let n = n.min(self.capacity);
        (n > 0 && bytes > 0)
            .then(|| crate::gpu::readback::ReadbackTicket::request(&self.buf_extra_readback, bytes))
    }

    /// Decodes a mapping requested by [`Self::request_read`].
    pub fn collect_read(
        &self,
        n: usize,
        bytes: u64,
        ticket: crate::gpu::readback::ReadbackTicket,
    ) -> Vec<[f32; EXTRA_OUT_STRIDE]> {
        let n = n.min(self.capacity);
        if n == 0 || bytes == 0 || !ticket.is_ready() {
            return Vec::new();
        }

        let slice = self.buf_extra_readback.slice(0..bytes);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let mut out = vec![[0.0f32; EXTRA_OUT_STRIDE]; n];
        for i in 0..n {
            let base = i * EXTRA_OUT_STRIDE;
            out[i].copy_from_slice(&floats[base..base + EXTRA_OUT_STRIDE]);
        }
        drop(data);
        self.buf_extra_readback.unmap();
        out
    }

    /// Reads `extra_out` back after submission.
    /// of the encoder that contained `bind_and_dispatch`. Same pattern as
    /// `PerceptionGpu::read_after_submit` (map_async → poll(Wait) → recv — NEVER in
    /// the reverse order, that is the deadlock documented twice in this codebase).
    ///
    /// Returns `[min_wall_ahead, edge_row6, edge_row7]` per being.
    pub fn read_after_submit(
        &self,
        device: &wgpu::Device,
        n: usize,
        bytes: u64,
    ) -> Vec<[f32; EXTRA_OUT_STRIDE]> {
        let Some(ticket) = self.request_read(n, bytes) else {
            return Vec::new();
        };
        device.poll(wgpu::Maintain::Wait);
        self.collect_read(n, bytes, ticket)
    }

    /// Zeroes the recurrent CfC state for the first `n` beings.
    ///
    /// Called between generations (same spot where `context.rs::reset_cfc` used to recreate
    /// `RetinaPerception::new()` per being). Uses `write_buffer` — no encoder, same
    /// pattern as `BrainGpu::reset_cfc`. CRITICAL: forgetting this reintroduces the bug
    /// documented on 2026-08-11 (child inheriting the parent's CfC state, non-deterministic
    /// elite re-evaluation).
    pub fn reset(&self, queue: &wgpu::Queue, n: usize) {
        let n = n.min(self.capacity);
        if n == 0 {
            return;
        }
        let zeros: Vec<u8> = vec![0u8; n * CFC_HIDDEN * std::mem::size_of::<f32>()];
        queue.write_buffer(&self.buf_cfc_state, 0, &zeros);
    }

    /// Clears only one being's perception digest CfC state.
    ///
    /// The ecology runtime calls this when a dead slot receives a newborn; the other
    /// beings' recurrent perception remains resident and untouched.
    pub fn reset_slot(&self, queue: &wgpu::Queue, being: usize) {
        if being >= self.capacity {
            return;
        }
        let bytes_per_being = CFC_HIDDEN * std::mem::size_of::<f32>();
        let zeros = vec![0u8; bytes_per_being];
        queue.write_buffer(
            &self.buf_cfc_state,
            (being * bytes_per_being) as u64,
            &zeros,
        );
    }
}

impl RetinaDigestGpu {
    pub fn snapshot_buffers(&self) -> Vec<crate::gpu::state_snapshot::NamedBuffer<'_>> {
        vec![
            ("retina_digest.cfc_state", &self.buf_cfc_state),
            ("retina_digest.extra_out", &self.buf_extra_out),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `EXTRA_OUT_STRIDE` is the contract with `context.rs` — changing it without updating
    /// both sides breaks silently (reads garbage from the next being). Locks the value here.
    /// P1.8: 3→19 (16 new slots for the CfC state that stopped aliasing into
    /// `buf_output` — see the constant's doc).
    #[test]
    fn extra_out_stride_e_dezenove() {
        assert_eq!(EXTRA_OUT_STRIDE, 19);
    }
}

// ─── Part 2: CPU vs GPU parity — marked #[ignore] (see the `retina_digest_parity` module) ──
#[cfg(test)]
#[path = "retina_digest_parity_test.rs"]
mod retina_digest_parity_test;
