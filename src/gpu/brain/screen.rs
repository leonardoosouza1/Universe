//! `BrainGpu`'s Task 5 screen constructors: `new_for_genomes` and `step_from_wave`.
//!
//! Split out of `brain.rs` proper (Task 5 review round, minor finding) rather than
//! left inline: `brain.rs` was already well over `docs/STYLE.md`'s 600-line cap before
//! this task touched it (1333 lines, itself already on the split queue), and these two
//! methods have nothing to do with the tick-time dispatch path the rest of the file is
//! about — they exist to open a STANDALONE device for the screen binary, which
//! `gpu_evolve`/`gpu_loop`'s shared-device `BrainGpu` never needs. A child module of
//! `brain` keeps private-field access (`self.device`, `self.queue`, `self.buf_wave`,
//! ...) without making any of it `pub(crate)`.

use std::sync::Arc;

use crate::config::iara::brain::SENSORY_WAVE_SIZE;
use crate::engine::SovereignEngine;
use crate::gpu::evolution::Genome;

use super::BrainGpu;

impl BrainGpu {
    // ── SCREEN (Task 5): standalone construction from genomes ──────────────

    /// Builds a brain context sized for exactly `genomes.len()` beings and uploads
    /// every genome's weights, τ bank and P1.7 neuron metadata — the three
    /// per-generation uploads `gpu_evolve/main.rs` performs before a generation runs
    /// (`upload_weights`, `upload_tau`, `upload_neuron_meta`; see `main.rs:2473-2488`).
    ///
    /// Batched (Task 5 review, C2): the screen scores many candidates against the
    /// SAME recorded wave per tick, so one dispatch of `genomes.len()` beings replaces
    /// `genomes.len()` sequential single-being dispatches — each of which pays its own
    /// `poll(Wait)` stall twice (`bind_and_dispatch`'s copy and `read_intent_after_submit`).
    ///
    /// Opens its own `wgpu` device instead of sharing one, the same pattern
    /// `gpu_validate.rs::validate_brain` uses: the screen has no simulation running
    /// alongside it to share a device with. `device_queue()` exposes the `Arc`s so a
    /// caller building companion GPU state (`ScreenBrain`'s slow brain, for the M4
    /// region gates) shares this exact device rather than opening a second one.
    ///
    /// `out_beta`/`out_gain` are left at `OUT_BETA_DEFAULT`/`OUT_GAIN_DEFAULT` — today's
    /// `gpu_evolve` defaults. `norm_l2_by_std` is forced to `true`: `new` itself defaults
    /// it to `false`, but `gpu_evolve/main.rs:685` has overridden it to `true`
    /// unconditionally since P1.8, so `true` is the value any current recording was
    /// actually made under. A caller screening a recording made under a *different*
    /// override reads the answer from the recording's own `wave_record::WaveConfig`
    /// header now (`--out-beta` guessing on the screen's own CLI is gone) and sets the
    /// matching `pub` field on the returned `BrainGpu` before calling `step_from_wave`.
    ///
    /// M4 region gates (`buf_gates`) stay at their neutral default (1.0, set in `new`)
    /// — this constructor does not touch them. Reproducing the M4 conductor is
    /// `ScreenBrain`'s job (`gpu::screen_brain`), not this kernel-level constructor's:
    /// it needs the slow brain plus a per-tick CPU reserve integral
    /// (`gpu_evolve/context.rs:1512-1540`), a second GPU subsystem outside the brain
    /// kernel this type wraps.
    pub fn new_for_genomes(genomes: &[Genome]) -> Result<Self, String> {
        let n = genomes.len();
        if n == 0 {
            return Err("new_for_genomes: no genomes given".to_string());
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| {
            "no Vulkan adapter found. On the RX 6750 XT this usually means the gfx override \
             is missing: export HSA_OVERRIDE_GFX_VERSION=10.3.0 and retry. Check \
             `vulkaninfo --summary` for a visible device."
                .to_string()
        })?;

        // 13 storage bindings (1..=13) + 1 uniform, the same layout `bind_and_dispatch`
        // uses in production. 16 mirrors `gpu_evolve/context.rs:269` (the limit the
        // brain kernel already runs under there) rather than `gpu_validate.rs`'s 12,
        // which predates `buf_gates`/`buf_neuron_mem` and is one short of what `new`
        // actually binds today.
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 16;
        let (device_raw, queue_raw) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("brain-screen"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("request_device: {e}"))?;

        let adapter_name = adapter.get_info().name.clone();
        let device = Arc::new(device_raw);
        let queue = Arc::new(queue_raw);
        let mut brain = Self::new(Arc::clone(&device), Arc::clone(&queue), adapter_name, n);
        brain.norm_l2_by_std = true;

        let mut engines: Vec<SovereignEngine> = Vec::with_capacity(n);
        for g in genomes {
            let mut eng = SovereignEngine::new();
            g.apply_to_engine(&mut eng);
            engines.push(eng);
        }
        let refs: Vec<&SovereignEngine> = engines.iter().collect();
        brain.upload_weights(&refs);
        brain.upload_tau(&Genome::pack_taus(genomes), n);
        brain.upload_neuron_meta(&Genome::pack_neuron_meta(genomes), n);

        Ok(brain)
    }

    /// `Arc` clones of the device/queue this context opened — lets a caller build a
    /// companion GPU context (e.g. `SlowBrainGpu`) on the SAME device instead of
    /// opening a second one, the same reason `RetinaGpu`/`BrainGpu` share one in
    /// production (see this file's module doc, "Why `Arc` and not references?").
    pub fn device_queue(&self) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        (Arc::clone(&self.device), Arc::clone(&self.queue))
    }

    /// Runs one tick of the brain kernel for all `n` beings this context was built
    /// for, replaying the SAME sensory wave into every one of them, and returns each
    /// being's three motor commands `[throttle, steer, brake]` —
    /// `final_intent[0..3)`, the same slots `gpu_evolve/context.rs` reads to record
    /// the wave (see the comment above `wave_writer` there).
    ///
    /// `wave` must hold `SENSORY_WAVE_SIZE` (192) values, each an i8 channel value
    /// already widened to f32 — the layout `wave_record::read_all` returns. `dt` is the
    /// tick's time delta in seconds — read it from the recording's `WaveConfig`, do not
    /// assume 1/64 (that assumption, baked into the screen's CLI rather than read from
    /// the recording, is what the Task 5 review's I3 finding removed). `n` must equal
    /// the number of beings this context was built with (`new_for_genomes(genomes).len()`).
    ///
    /// Uses the CPU-wave path of `bind_and_dispatch` (`buf_percept = None`), the same
    /// one `gpu_validate.rs` exercises: it uploads `n` replicas of `wave` into
    /// `buf_wave` and never touches the GPU perception buffer, which this screen never
    /// populates.
    pub fn step_from_wave(&self, wave: &[f32], dt: f32, n: usize) -> Vec<[f32; 3]> {
        assert_eq!(
            wave.len(),
            SENSORY_WAVE_SIZE,
            "wave has {} channels, expected {SENSORY_WAVE_SIZE}",
            wave.len()
        );
        // Each value is an i8 channel widened to f32 by `wave_record` (and, before
        // that, by the recorder in `context.rs`) — casting back to i32 is exact, not a
        // rounding operation. Replicated `n` times: every being in this batch is a
        // different CANDIDATE GENOME scored against the same recorded tick.
        let mut wave_i32: Vec<i32> = Vec::with_capacity(n * SENSORY_WAVE_SIZE);
        for _ in 0..n {
            wave_i32.extend(wave.iter().map(|&v| v as i32));
        }

        let mut enc = self.device.create_command_encoder(&Default::default());
        self.bind_and_dispatch(&mut enc, n, &wave_i32, dt, None);
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        self.read_intent_after_submit(n)
            .into_iter()
            .map(|full| [full[0], full[1], full[2]])
            .collect()
    }
}
