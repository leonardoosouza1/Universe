//! `ScreenBrain` — Task 5's screen, batched across K candidate genomes, replaying one
//! recorded wave through the fast brain kernel AND the M4 region-gate machinery (slow
//! brain + CPU reserve integral) together.
//!
//! # Why the slow brain is here at all
//!
//! The Task 5 review (finding C1) measured the first cut's parity floor — 0.053, flat
//! across ticks 5000-14000 rather than decaying — and pointed at `buf_gates` being
//! left neutral. `--neuromod-regions` was on for the recording this screen calibrates
//! against, and `run.log`'s own end-of-generation telemetry puts the gates within
//! roughly ±16% of neutral (backed out from the reserve-depleted values printed at
//! `gpu_evolve/main.rs:3631`), so leaving the gate at 1.0 was always scoring an
//! approximation, not the recording — reproducing it is correct regardless of how much
//! of the floor it explains.
//!
//! In the end M4 alone only closed part of the gap (0.053 → 0.047): the flat, non-
//! decaying floor's larger cause turned out to be a standing P1.7 period-counter phase
//! offset from skipping row 0 entirely, fixed separately in `bin/screen.rs` by
//! replaying row 0 as a genuine warm-up. Both fixes landed together and parity is now
//! exact (0.0000) — see the Task 5 fix report for the combined measurement.
//!
//! Reproducing it needs the slow brain (`SlowBrainGpu`) because the gate's raw value
//! is `1 + 0.5*tanh(w·slow_out + b)` — `slow_out` is the slow brain's own forward
//! pass, computed from the same wave this screen already replays. Everything below is
//! either a call into `BrainGpu`/`SlowBrainGpu`'s existing public API, or a direct port
//! of the ~15-line CPU reserve integral `gpu_evolve/context.rs:1512-1540` already runs
//! — no layer is reimplemented, only orchestrated from the outside.
//!
//! # Batching (Task 5 review, finding C2)
//!
//! One genome per dispatch was 20 000 sequential single-being GPU round trips — ~3.4x
//! SLOWER per genome than `gpu_evolve`'s own 48-being closed loop, because each
//! dispatch pays two blocking `poll(Wait)` stalls regardless of how much work it does.
//! `ScreenBrain::new_for_genomes` builds one context for K genomes and `step` replays
//! the SAME wave into all K of them in one dispatch — the wave differs only per TICK,
//! never per candidate, which is exactly what the fast brain's `n` beings and the slow
//! brain's `n` beings already parallelise for free.
//!
//! # The one-tick gate lag
//!
//! In `context.rs`'s own tick order, the fast brain fires FIRST (using whatever gate
//! the PREVIOUS tick's slow-brain output produced) and the slow brain runs AFTER (its
//! output becomes the gate for the NEXT tick). `step` reproduces that lag exactly:
//! `self.gate` holds the value applied on THIS call's fast-brain dispatch, and is
//! updated at the end of the SAME call from THIS tick's `slow_out`, for use on the
//! NEXT call.
//!
//! `ScreenBrain` itself always starts at the neutral gate `[1.0; 4]` — the correct
//! state for a freshly spawned being's FIRST EVER dispatch, which is exactly what this
//! struct's first `step` call is. It is `bin/screen.rs`'s job to make sure that first
//! call is the recording's row 0 (the JIT warm-up tick), not row 1: replaying row 0
//! first — discarding its own prediction, keeping everything it advances (CfC state,
//! the P1.7 period counter, and this gate/reserve pair) — is what makes row 1 onward
//! line up with the recording exactly. Calling `step` with row 1 first would silently
//! score every genome against gates the true recording never applied to row 1.

use std::sync::Arc;

use crate::engine::SovereignEngine;
use crate::gpu::brain::BrainGpu;
use crate::gpu::evolution::Genome;
use crate::gpu::slow_brain::SlowBrainGpu;
use crate::gpu::wave_record::{pack_wave_u32, WAVE_U32S};

/// Length of `Genome::gates_w` when the M4 conductor block is present on disk (4
/// gates × (3 slow-brain outputs + bias) = 16 — see `Genome::gates_w`'s own doc).
const GATES_W_LEN: usize = 16;

/// Fast brain + slow brain + M4 reserve integral, batched across `n` candidate
/// genomes replaying the SAME recorded wave.
pub struct ScreenBrain {
    /// `pub` so `bin/screen.rs` can read `out_beta`/`out_gain`/etc. off it, exactly as
    /// it would on a bare `BrainGpu` — batching and M4 do not change that contract.
    pub brain: BrainGpu,
    slow: SlowBrainGpu,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    wave_packed_buf: wgpu::Buffer,
    n: usize,
    /// `n * GATES_W_LEN` f32 — genome `i`'s M4 conductor weights, or a zero row if that
    /// genome carries no `gates_w` block (legacy genome, mechanism never trained).
    gates_w: Vec<f32>,
    /// Per-genome burnout reserve, `context.rs`'s `nm_reserva`. Starts `[1.0; 4]`, the
    /// same spawn default `main.rs:302,2648` use.
    reserve: Vec<[f32; 4]>,
    /// Per-genome gate APPLIED on the next `step` call's fast-brain dispatch. Starts
    /// `[1.0; 4]` (neutral) — see the module doc's note on the one-tick lag.
    gate: Vec<[f32; 4]>,
    /// Per-genome: run the M4 integral for it at all? False when either
    /// `--neuromod-regions` was off for the recording being screened, or this genome
    /// has no `gates_w` (mirrors `context.rs`'s own
    /// `NEUROMOD_REGIOES.load() && !gates_w.is_empty()` gate).
    active: Vec<bool>,
}

impl ScreenBrain {
    /// Builds a batch of `genomes.len()` candidates. `neuromod_regioes` should come
    /// straight from the recording's `wave_record::WaveConfig::neuromod_regioes` — the
    /// screen has no business deciding this on its own, the recording already says
    /// whether the champion that produced it ran with the mechanism on.
    ///
    /// Errors if any genome's `gates_w` is a length other than 0 (no M4 block) or 16
    /// (`GATES_W_LEN`) — anything else is a corrupt or foreign file, not a legitimate
    /// "mechanism absent" case, and silently zero-padding it would misreport which
    /// genomes are actually gate-driven.
    pub fn new_for_genomes(genomes: &[Genome], neuromod_regioes: bool) -> Result<Self, String> {
        let n = genomes.len();
        let brain = BrainGpu::new_for_genomes(genomes)?;
        let (device, queue) = brain.device_queue();
        let slow = SlowBrainGpu::new_shared(Arc::clone(&device), Arc::clone(&queue), n);

        let mut slow_weights = Vec::with_capacity(n * 1091);
        let mut gates_w = Vec::with_capacity(n * GATES_W_LEN);
        let mut active = Vec::with_capacity(n);
        for (i, g) in genomes.iter().enumerate() {
            // Reuses `apply_to_engine`'s own "empty `slow` -> `SlowBrain::new()`'s
            // fixed-seed default" fallback instead of re-deciding what an absent slow
            // brain means here — one source of truth for that rule.
            let mut eng = SovereignEngine::new();
            g.apply_to_engine(&mut eng);
            slow_weights.extend_from_slice(&eng.slow_brain.all_weights());

            match g.gates_w.len() {
                0 => {
                    gates_w.extend(std::iter::repeat_n(0.0f32, GATES_W_LEN));
                    active.push(false);
                }
                GATES_W_LEN => {
                    gates_w.extend_from_slice(&g.gates_w);
                    active.push(neuromod_regioes);
                }
                other => {
                    return Err(format!(
                        "genome {i}: gates_w has {other} values, expected 0 (no M4 \
                         block) or {GATES_W_LEN} (the M4 block) — corrupt or foreign \
                         genome file."
                    ));
                }
            }
        }
        slow.upload_weights(&slow_weights, n);

        let wave_packed_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen-wave-packed"),
            size: (n * WAVE_U32S * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            brain,
            slow,
            device,
            queue,
            wave_packed_buf,
            n,
            gates_w,
            reserve: vec![[1.0; 4]; n],
            gate: vec![[1.0; 4]; n],
            active,
        })
    }

    /// Number of genomes in this batch.
    pub fn len(&self) -> usize {
        self.n
    }

    /// `new_for_genomes` rejects an empty slice, so a live `ScreenBrain` is never
    /// empty — this exists only to satisfy `clippy::len_without_is_empty`.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Runs one tick for every genome in the batch against the SAME recorded wave, and
    /// returns each genome's `[throttle, steer, brake]`. See the module doc for the
    /// tick order this reproduces (fast brain first, on the PREVIOUS tick's gate; slow
    /// brain after, producing the NEXT tick's gate).
    pub fn step(&mut self, wave: &[f32], dt: f32) -> Vec<[f32; 3]> {
        let n = self.n;

        // ── 1. Fast brain: this tick's prediction, using the gate `step`'s last call
        //    (or construction, for the very first call) computed. ──────────────────
        let flat_gate: Vec<f32> = self.gate.iter().flatten().copied().collect();
        self.brain.upload_gates(&flat_gate, n);
        let pred = self.brain.step_from_wave(wave, dt, n);

        // ── 2. Slow brain: gathers 48 channels straight from a GPU buffer, exactly
        //    the layout `dispatch_gather` already expects (`slow_gather.wgsl`), so no
        //    CPU reimplementation of the channel map is needed. Skipped entirely when
        //    no genome in the batch is gate-driven — matches `context.rs`'s own
        //    `NEUROMOD_REGIOES && !gates_w.is_empty()` short-circuit. ────────────────
        if self.active.iter().any(|&a| a) {
            let packed = pack_wave_u32(wave, n);
            self.queue
                .write_buffer(&self.wave_packed_buf, 0, bytemuck::cast_slice(&packed));

            let mut enc = self.device.create_command_encoder(&Default::default());
            self.slow
                .dispatch_gather(&mut enc, &self.wave_packed_buf, n);
            self.slow.dispatch(&mut enc, n, dt);
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            let slow_out = self.slow.read_outputs(n);

            // ── M4 reserve integral — direct port of `context.rs:1512-1540`. `dt` is
            //    the recording's own tick length (passed in), not a re-hardcoded 1/64:
            //    production hardcodes `dt_nm = 1.0/64.0` there because `gpu_evolve`
            //    itself never runs at any other rate, but this screen has no business
            //    assuming that a SECOND time when the caller already has the true `dt`
            //    from the recording's `WaveConfig`. ─────────────────────────────────
            for i in 0..n {
                if !self.active[i] {
                    continue; // gate stays neutral forever for this genome
                }
                let so = slow_out.get(i).copied().unwrap_or([0.0; 3]);
                let w = &self.gates_w[i * GATES_W_LEN..(i + 1) * GATES_W_LEN];
                let mut new_gate = [0.0f32; 4];
                for b in 0..4 {
                    let z = w[b * 4] * so[0]
                        + w[b * 4 + 1] * so[1]
                        + w[b * 4 + 2] * so[2]
                        + w[b * 4 + 3];
                    let g_raw = 1.0 + 0.5 * z.tanh();
                    let r = &mut self.reserve[i][b];
                    let uso = (g_raw - 1.0).abs();
                    *r = (*r + ((1.0 - *r) / 30.0 - 0.5 * uso * *r) * dt).clamp(0.0, 1.0);
                    new_gate[b] = 1.0 + (g_raw - 1.0) * *r;
                }
                self.gate[i] = new_gate;
            }
        }

        pred
    }
}
