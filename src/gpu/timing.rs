//! Per-kernel GPU timing.
//!
//! Why this exists: on 2026-08-26 `rocprofv3` produced an EMPTY trace against this
//! binary. wgpu runs on Vulkan/RADV and rocprof instruments HIP/HSA, so no external
//! profiler can see these kernels. Without this module every optimisation is a guess.
//! The `timestamp_writes` fields were already present on every compute pass, set to
//! `None` (`core.rs:503`, `cortex_sheet.rs:521`, `retina_digest.rs:249`).
//!
//! Two ways to place a timestamp, both writing into the same `QuerySet`: `scope`
//! hands `ComputePassTimestampWrites` to `begin_compute_pass`, timing exactly one
//! pass — the only option when the pass is opened where the timer cannot reach
//! (`gpu_evolve`'s first caller found several: `PerceptionGpu::bind_and_dispatch`,
//! `BrainGpu::bind_and_dispatch`, `SlowBrainGpu::dispatch`/`dispatch_gather` all open
//! their passes inside shared library methods called from several binaries, and
//! threading a timer parameter through all of them would touch every caller).
//! `mark_begin`/`mark_end` instead call `CommandEncoder::write_timestamp` directly
//! from the call site that already has `&mut CommandEncoder` in hand, bracketing
//! the whole library call — coarser (no visibility into the sub-passes it contains)
//! but reachable without changing any shared signature, and it also captures the gap
//! BETWEEN consecutive dispatches that a per-pass `scope` hides.

/// Milliseconds between two GPU timestamps.
///
/// `period_ns` is `Queue::get_timestamp_period()`, nanoseconds per tick. Returns 0.0
/// when the counter went backwards (wrap) rather than a negative duration.
pub fn ticks_to_ms(begin: u64, end: u64, period_ns: f32) -> f32 {
    if end <= begin {
        return 0.0;
    }
    (end - begin) as f32 * period_ns / 1.0e6
}

/// Owns one `QuerySet` and the buffers to get its values back to the CPU.
///
/// Two timestamps per scope (begin, end). `max_passes` is the number of scopes per
/// frame; the set is allocated once and reused.
pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    labels: Vec<&'static str>,
    max_passes: usize,
    /// Whether the device also granted `Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`,
    /// checked once here at construction. `scope` never needs it (it writes through
    /// `ComputePassTimestampWrites`, gated by plain `TIMESTAMP_QUERY` alone); only
    /// `mark_begin`/`mark_end` do, since they call `CommandEncoder::write_timestamp`
    /// directly, outside any pass.
    can_mark_in_encoder: bool,
}

impl GpuTimer {
    /// `None` when the device was created without `Features::TIMESTAMP_QUERY`.
    pub fn new(device: &wgpu::Device, max_passes: usize) -> Option<GpuTimer> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let count = (max_passes * 2) as u32;
        let bytes = (count as u64) * 8;
        Some(GpuTimer {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("kernel-timing"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timing-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timing-read"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            labels: Vec::with_capacity(max_passes),
            max_passes,
            can_mark_in_encoder: device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        })
    }

    /// Reserves the next label slot and writes its "begin" timestamp directly on
    /// `enc`, outside any compute pass.
    ///
    /// Why this exists next to `scope`: `scope` times one pass (`SetPipeline` through
    /// the end of that `ComputePass`), so the gap BETWEEN two consecutive dispatches —
    /// driver overhead, a barrier, CPU-side bind-group setup recorded into the same
    /// encoder — falls outside every scope and is invisible. Bracketing the whole call
    /// (e.g. one `PerceptionGpu::bind_and_dispatch`, which opens several passes of its
    /// own) from the outside captures that gap as part of the measurement instead of
    /// hiding it, at the cost of losing visibility into the sub-passes it contains.
    ///
    /// Returns `false` — writing nothing — when the device lacks
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS` or `max_passes` slots are already used this
    /// frame. The caller must feed that value into the matching `mark_end`: an
    /// unconditional `mark_end` after a dropped `mark_begin` would write an "end"
    /// timestamp into whatever slot happens to be last, corrupting an unrelated label.
    #[must_use]
    pub fn mark_begin(&mut self, enc: &mut wgpu::CommandEncoder, label: &'static str) -> bool {
        if !self.can_mark_in_encoder || self.labels.len() >= self.max_passes {
            return false;
        }
        let i = self.labels.len() as u32;
        self.labels.push(label);
        enc.write_timestamp(&self.set, i * 2);
        true
    }

    /// Writes the "end" timestamp paired with the most recent successful
    /// `mark_begin`. `began` must be the value that call returned; `false` is a no-op,
    /// so a skipped begin never writes a mismatched end.
    pub fn mark_end(&mut self, enc: &mut wgpu::CommandEncoder, began: bool) {
        if !began {
            return;
        }
        let i = (self.labels.len() - 1) as u32;
        enc.write_timestamp(&self.set, i * 2 + 1);
    }

    /// Call once per frame before recording passes.
    pub fn begin_frame(&mut self) {
        self.labels.clear();
    }

    /// Returns the `timestamp_writes` to hand to `begin_compute_pass`.
    /// Silently returns `None` past `max_passes` rather than panicking mid-run.
    pub fn scope(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if self.labels.len() >= self.max_passes {
            return None;
        }
        let i = self.labels.len() as u32;
        self.labels.push(label);
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(i * 2),
            end_of_pass_write_index: Some(i * 2 + 1),
        })
    }

    /// Record the resolve + copy. Must be on the same encoder as the passes.
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder) {
        let n = (self.labels.len() * 2) as u32;
        if n == 0 {
            return;
        }
        enc.resolve_query_set(&self.set, 0..n, &self.resolve_buf, 0);
        enc.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.read_buf, 0, (n as u64) * 8);
    }

    /// Read back after the submit has been polled. One entry per scope, in order.
    ///
    /// Follows this repository's documented readback order, `map_async` → `poll` →
    /// `recv` (`src/gpu/CLAUDE.md:86`): polling before `map_async` is a silent
    /// deadlock, already paid for once in `brain.rs:1196-1199` (a poll issued before
    /// the map request never serves the callback, so `recv()` blocks forever with no
    /// error).
    ///
    /// Unlike the readbacks this pattern is copied from (`core.rs:549-555`,
    /// `brain.rs:696-700`), which forward the map result through the channel and then
    /// discard it with `let _ = rx.recv();`, this one inspects it. A timing readback is
    /// diagnostic-only — it must never crash a race over a missing frame of
    /// measurements — so a mapping failure (device lost, buffer already mapped, a
    /// driver hiccup) is reported with `eprintln!` and an empty `Vec` instead of
    /// surfacing as an opaque panic inside `get_mapped_range()`.
    pub fn read_ms(&self, device: &wgpu::Device, period_ns: f32) -> Vec<(&'static str, f32)> {
        let n = self.labels.len();
        if n == 0 {
            return Vec::new();
        }
        let slice = self.read_buf.slice(..(n as u64) * 16);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!(
                    "GpuTimer::read_ms: buffer mapping failed ({e:?}) — no timings this frame"
                );
                return Vec::new();
            }
            Err(_) => {
                eprintln!(
                    "GpuTimer::read_ms: mapping callback channel closed before responding \
                     (GPU likely lost) — no timings this frame"
                );
                return Vec::new();
            }
        }
        let data = slice.get_mapped_range();
        let raw: &[u64] = bytemuck::cast_slice(&data);
        let out = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| (*l, ticks_to_ms(raw[i * 2], raw[i * 2 + 1], period_ns)))
            .collect();
        drop(data);
        self.read_buf.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::ticks_to_ms;

    /// The timestamp period is nanoseconds per tick and comes from the queue.
    /// RADV reports 1.0 on this card; a wrong period silently scales every
    /// measurement, which is the failure mode that makes a profiler lie.
    #[test]
    fn converts_timestamp_ticks_to_milliseconds() {
        // 1_000_000 ticks at 1 ns each = 1 ms.
        assert!((ticks_to_ms(0, 1_000_000, 1.0) - 1.0).abs() < 1e-6);
        // A 40 ns period makes the same span 40 ms.
        assert!((ticks_to_ms(0, 1_000_000, 40.0) - 40.0).abs() < 1e-3);
        // Counters wrap; a smaller end must not produce a negative duration.
        assert_eq!(ticks_to_ms(500, 100, 1.0), 0.0);
    }
}
