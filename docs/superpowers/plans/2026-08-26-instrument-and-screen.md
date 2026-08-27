# Phase 0 (kernel instrument) + Phase 2 (open-loop screen) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine measurable per kernel, and make candidate brains rejectable in milliseconds instead of seconds.

**Architecture:** Two independent tracks. Phase 0 turns on wgpu timestamp queries around the compute passes that already exist, producing a per-kernel millisecond table — today the project has no per-kernel timing at all, because rocprof cannot see Vulkan. Phase 2 records the raw sensory wave once, then replays it through the EXISTING brain kernel for many candidate genomes at once, scoring each against the champion's recorded commands. Phase 2 deliberately reuses the GPU brain rather than reimplementing it on the CPU: this repository has already shipped two disjoint brains once (`project_dois_cerebros_disjuntos`), and a CPU replica would reintroduce that failure.

**Tech Stack:** Rust 2024 (MSRV 1.85), wgpu on Vulkan/RADV, WGSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-26-gpu-resident-scenarios-design.md`

## Global Constraints

- **English only** for every new or touched line — identifiers, comments, log and panic messages, test names, commit messages. Gate: `BASE_REF=<base> bash scripts/check_language.sh`.
- **`cargo fmt -p universe --check` must pass.** Never `cargo fmt --all` — it rewrites `vendor/` and `scripts/check_vendor.sh` must keep reporting 5/5 identical.
- **`cargo test --release` must stay green.** Baseline is 408 tests as of commit `9e323a2`.
- **No GPU training run.** `--gens 1` evaluation only, until the owner explicitly releases training.
- **New flags are off by default**, and with the flag off `obs.f32` must be byte-identical to a binary built before the change (same sha256). This check caught three real problems on 2026-08-26.
- **Files stay under ~600 lines** (`docs/STYLE.md`). `src/bin/gpu_evolve/main.rs` is already ~2600 lines; do not grow it beyond flag parsing and a call.
- **Heavy outputs never in the repository** — recordings go to `/media/llm-data/`. Check `df -h /media/llm-data` before writing anything large.
- Every performance claim in a commit message carries the command that produced it.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/gpu/timing.rs` (create) | `GpuTimer`: owns the `QuerySet`, resolve buffer and readback; converts ticks to milliseconds. Nothing else knows about queries. |
| `src/gpu/mod.rs` (modify, ~line 208) | Request `TIMESTAMP_QUERY` when the adapter offers it; expose whether it was granted. |
| `src/bin/gpu_evolve/context.rs` (modify, ~line 1290-1303) | Pass a `Option<&mut GpuTimer>` through the tick so passes can be labelled. |
| `src/gpu/wave_record.rs` (create) | Writes and reads the raw sensory-wave recording. One format, one place. |
| `src/bin/screen.rs` (create) | The screening binary: load recording + genomes, replay through the brain kernel, report per-tick error. |

---

## Task 1: `GpuTimer` — the instrument, standalone and tested

**Files:**
- Create: `src/gpu/timing.rs`
- Modify: `src/gpu/mod.rs` (add `pub mod timing;`)
- Test: inside `src/gpu/timing.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub struct GpuTimer`
  - `pub fn new(device: &wgpu::Device, max_passes: usize) -> Option<GpuTimer>` — `None` when the feature is absent.
  - `pub fn begin_frame(&mut self)` — call once per frame before recording passes
  - `pub fn scope(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>>` — `None` past `max_passes`
  - `pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder)`
  - `pub fn read_ms(&self, device: &wgpu::Device, period_ns: f32) -> Vec<(&'static str, f32)>`
  - `pub fn ticks_to_ms(begin: u64, end: u64, period_ns: f32) -> f32`

- [ ] **Step 1: Write the failing test for the pure conversion**

Only the arithmetic is unit-testable without a device; the GPU path is covered by Task 3's acceptance check.

```rust
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
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --lib timing 2>&1 | tail -20`
Expected: FAIL — `cannot find function ticks_to_ms` / unresolved module.

- [ ] **Step 3: Write the minimal module**

```rust
//! Per-kernel GPU timing.
//!
//! Why this exists: on 2026-08-26 `rocprofv3` produced an EMPTY trace against this
//! binary. wgpu runs on Vulkan/RADV and rocprof instruments HIP/HSA, so no external
//! profiler can see these kernels. Without this module every optimisation is a guess.
//! The `timestamp_writes` fields were already present on every compute pass, set to
//! `None` (`core.rs:503`, `cortex_sheet.rs:521`, `retina_digest.rs:249`).

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
```

- [ ] **Step 4: Run it and watch it pass**

Run: `cargo test --release --lib timing 2>&1 | grep "test result"`
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Add the query-set plumbing (no test — Task 3 exercises it)**

Append to `src/gpu/timing.rs`:

```rust
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
        })
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
    pub fn read_ms(&self, device: &wgpu::Device, period_ns: f32) -> Vec<(&'static str, f32)> {
        let n = self.labels.len();
        if n == 0 {
            return Vec::new();
        }
        let slice = self.read_buf.slice(..(n as u64) * 16);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
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
```

Add `pub mod timing;` to `src/gpu/mod.rs` next to the other `pub mod` lines.

- [ ] **Step 6: Build and run the full suite**

Run: `cargo build --release 2>&1 | grep -E "^error" -A3; cargo test --release 2>&1 | grep -E "^test result: FAILED|failures:"`
Expected: no errors, no failures.

- [ ] **Step 7: Commit**

```bash
cargo fmt -p universe
git add src/gpu/timing.rs src/gpu/mod.rs
git commit -m "feat(gpu): a GpuTimer, because no external profiler can see these kernels

rocprofv3 returns an empty trace against this binary: wgpu runs on Vulkan/RADV and
rocprof instruments HIP/HSA. The wgpu timestamp path was present and unused
(timestamp_writes: None at core.rs:503, cortex_sheet.rs:521, retina_digest.rs:249).

This adds the instrument only; nothing is wired to it yet. ticks_to_ms is unit
tested including the counter-wrap case, which would otherwise report a negative
duration.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Request the feature, and degrade honestly when it is absent

**Files:**
- Modify: `src/gpu/mod.rs:208-215` (the `request_device` call)
- Test: `tests/timing_feature.rs` (create)

**Interfaces:**
- Consumes: `GpuTimer::new` from Task 1.
- Produces: the device is created with `TIMESTAMP_QUERY` whenever the adapter supports it. No new public function; `device.features()` carries the answer.

- [ ] **Step 1: Write the failing test**

```rust
//! The feature must be REQUESTED when available and its absence must not be fatal.
//! `#[ignore]`d because it opens a real adapter: CI runners have none, and a test that
//! silently passes on a missing device is worse than one that is skipped.

#[test]
#[ignore = "opens a real GPU — run manually: cargo test --release --test timing_feature -- --ignored"]
fn timestamp_query_is_requested_when_the_adapter_offers_it() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no adapter");

    let offers = adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);

    let core = universe::gpu::GpuCore::new().expect("device");
    let granted = core
        .device()
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);

    assert_eq!(
        offers, granted,
        "adapter offers TIMESTAMP_QUERY = {offers} but the device was created with {granted}"
    );
}
```

Note: if `GpuCore` is not the type name or `device()` is not exposed, read `src/gpu/mod.rs` and use the real accessor — do not add a new one just for the test.

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --test timing_feature -- --ignored 2>&1 | tail -20`
Expected: FAIL — `offers true but granted false`.

- [ ] **Step 3: Request the feature conditionally**

In `src/gpu/mod.rs`, replace `required_features: wgpu::Features::empty(),` with:

```rust
// TIMESTAMP_QUERY is requested only when the adapter offers it: asking for a
// feature the adapter lacks makes request_device return None, and losing the
// whole device to gain a profiler is a bad trade. `GpuTimer::new` returns None
// when it was not granted, and the caller prints nothing rather than lying.
required_features: adapter_obj.features() & wgpu::Features::TIMESTAMP_QUERY,
```

- [ ] **Step 4: Run it and watch it pass**

Run: `cargo test --release --test timing_feature -- --ignored 2>&1 | grep "test result"`
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Prove the change is inert for normal runs**

The device now carries an extra feature; nothing may change numerically.

```bash
G=/media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin
D=/media/llm-data/scratch/tsq; rm -rf $D; mkdir -p $D; cp $G $D/
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 48 --ticks 20000 \
  --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
  --pistas 8005,8200,9001 --seed 77 --observar $D/obs.f32
sha256sum $D/obs.f32
```

Expected: median `2939m` and the same sha256 as the pre-change binary. If it differs, stop — requesting a feature changed behaviour, which is a driver-level surprise worth understanding before going further.

- [ ] **Step 6: Commit**

```bash
cargo fmt -p universe
git add src/gpu/mod.rs tests/timing_feature.rs
git commit -m "feat(gpu): request TIMESTAMP_QUERY when the adapter offers it

Masked with the adapter's own feature set rather than requested outright:
request_device returns None if a missing feature is required, and losing the device
to gain a profiler is a bad trade. GpuTimer::new already returns None when the
feature was not granted.

Verified inert: obs.f32 byte-identical to the pre-change binary on the standard
48-being 20k-tick evaluation, median 2939 m unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Wire the timer into the tick and print the table

**Files:**
- Modify: `src/bin/gpu_evolve/context.rs` (the tick, ~1290-1303)
- Modify: `src/bin/gpu_evolve/main.rs` (flag parsing only)

**Interfaces:**
- Consumes: `GpuTimer` (Task 1), the granted feature (Task 2).
- Produces: flag `--perfil-kernel <N>`; every N ticks, one line per kernel plus a `gaps` line.

- [ ] **Step 1: Add the flag, off by default**

In `main.rs`, next to the other flags:

```rust
"--perfil-kernel" => {
    i += 1;
    if i < args.len() {
        perfil_kernel = args[i].parse().unwrap_or(0);
    }
}
```

with `let mut perfil_kernel: u32 = 0;` alongside the other defaults, and pass it into the context.

- [ ] **Step 2: Label the passes in the tick**

For each `begin_compute_pass` reached during a tick, replace `timestamp_writes: None` with a scope from the timer, keeping `None` when the timer is absent:

```rust
timestamp_writes: timer.as_mut().and_then(|t| t.scope("perception.spatial")),
```

Labels to use, one per pass, in tick order: `retina`, `perception.spatial`, `perception.motion`, `perception.temporal`, `brain`, `slow.gather`, `slow`, `physics.a`, `physics.b`, `physics.c`. `GpuTimer::new(device, 16)` leaves headroom.

- [ ] **Step 3: Resolve and print**

Immediately before `self.queue.submit(...)` at `context.rs:1302`:

```rust
if let Some(t) = timer.as_mut() {
    t.resolve(&mut enc);
}
```

and after the existing `self.device.poll(wgpu::Maintain::Wait);`:

```rust
if let Some(t) = timer.as_ref() {
    if perfil_kernel > 0 && tick % perfil_kernel == 0 {
        let period = self.queue.get_timestamp_period();
        let rows = t.read_ms(&self.device, period);
        let soma: f32 = rows.iter().map(|(_, ms)| ms).sum();
        for (label, ms) in &rows {
            println!("  [perfil] {label:<22} {ms:8.4} ms");
        }
        println!("  [perfil] {:<22} {soma:8.4} ms (soma dos kernels)", "TOTAL");
    }
}
```

- [ ] **Step 4: Run it and read the table**

```bash
G=/media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin
D=/media/llm-data/scratch/perf; rm -rf $D; mkdir -p $D; cp $G $D/
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 768 --ticks 400 \
  --width 28 --out-beta 0.0 --pistas 8005 --seed 77 --perfil-kernel 100 2>&1 | grep perfil
```

- [ ] **Step 5: Check the instrument against the wall clock — THE acceptance gate**

The spec's acceptance: the parts plus the gaps must reconstruct wall-clock ms/tick within 10%. Measured wall clock at 768 beings is **33.61 ms/tick** (1500 ticks, `--sem-lento` off, brain on).

```bash
time ./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 768 --ticks 1500 \
  --width 28 --out-beta 0.0 --pistas 8005 --seed 77
```

Compare `TOTAL` from Step 4 against `wall / 1500 * 1000`. If the kernel sum is far below wall clock, the remainder is idle gaps and CPU work — **record that number explicitly**, because it is the answer to whether GPU-residency (Phase 1) is worth building.

- [ ] **Step 6: Verify the flag is inert when off**

```bash
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 48 --ticks 20000 \
  --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
  --pistas 8005,8200,9001 --seed 77 --observar $D/obs.f32
sha256sum $D/obs.f32
```

Expected: unchanged sha256, median `2939m`.

- [ ] **Step 7: Commit, with the measured table in the message**

```bash
cargo fmt -p universe
git add -A
git commit -F - <<'MSG'
feat(gpu_evolve): --perfil-kernel prints milliseconds per kernel

First per-kernel timing this project has ever had. Records the table measured at
768 beings, and the gap between the kernel sum and wall clock -- that gap is the
number that decides whether Phase 1 (GPU-resident state, K ticks per submit) is
worth building, and it was previously unknowable.

[paste the actual table here, plus wall-clock ms/tick and the reconstructed
percentage]

Off by default and verified inert: obs.f32 byte-identical on the standard
evaluation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
```

---

## Task 4: Record the raw sensory wave

**Files:**
- Create: `src/gpu/wave_record.rs`
- Modify: `src/bin/gpu_evolve/context.rs` (write one row per tick when enabled)
- Modify: `src/bin/gpu_evolve/main.rs` (flag)
- Test: inside `src/gpu/wave_record.rs`

**Interfaces:**
- Consumes: `percept.read_after_submit(&device, n, bytes)`, which `--auditar-visao` already uses.
- Produces:
  - `pub const WAVE_CHANNELS: usize = 192;`
  - `pub struct WaveRecord { pub ticks: usize }`
  - `pub fn write_row(w: &mut impl std::io::Write, wave: &[f32], intent: &[f32; 3]) -> std::io::Result<()>`
  - `pub fn read_all(path: &str) -> std::io::Result<(Vec<f32>, Vec<[f32; 3]>)>` — waves flattened row-major, one intent per row.
  - File format: `f32` little-endian, `195` per row = 192 wave channels + 3 recorded commands.

- [ ] **Step 1: Write the failing round-trip test**

```rust
#[cfg(test)]
mod tests {
    use super::{read_all, write_row, WAVE_CHANNELS};

    /// The format has silently dropped fields twice in this repository's history
    /// (the slow brain's 1091 weights, the 96-entry tau bank), so the round trip is
    /// tested before anything depends on it.
    #[test]
    fn a_recording_survives_the_disk() {
        let dir = std::env::temp_dir().join(format!("wave_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.f32");

        let wave_a: Vec<f32> = (0..WAVE_CHANNELS).map(|i| i as f32 * 0.5).collect();
        let wave_b: Vec<f32> = (0..WAVE_CHANNELS).map(|i| -(i as f32)).collect();
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            write_row(&mut f, &wave_a, &[0.1, -0.2, 0.3]).unwrap();
            write_row(&mut f, &wave_b, &[0.4, 0.5, -0.6]).unwrap();
        }

        let (waves, intents) = read_all(path.to_str().unwrap()).unwrap();
        assert_eq!(intents.len(), 2, "row count");
        assert_eq!(waves.len(), 2 * WAVE_CHANNELS, "wave payload size");
        assert_eq!(&waves[..WAVE_CHANNELS], &wave_a[..], "row 0 wave");
        assert_eq!(&waves[WAVE_CHANNELS..], &wave_b[..], "row 1 wave");
        assert_eq!(intents[1], [0.4, 0.5, -0.6], "row 1 command");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_short_read() {
        let dir = std::env::temp_dir().join(format!("wave_tr_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.f32");
        // 195 f32 per row; 100 f32 is half a row.
        std::fs::write(&path, vec![0u8; 100 * 4]).unwrap();
        assert!(read_all(path.to_str().unwrap()).is_err(), "must reject a partial row");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --lib wave_record 2>&1 | tail -20`
Expected: FAIL — unresolved module.

- [ ] **Step 3: Write the module**

```rust
//! Raw sensory-wave recording: the input side of the open-loop screen.
//!
//! `obs.f32` is NOT enough for this. Its columns 10..202 are post-KAN and post-CfC —
//! already transformed by the recording genome's own KAN knots and tau bank. Replaying
//! a DIFFERENT genome from them would hold those stages fixed and only vary L1/L2. The
//! raw 192-channel wave is the last point in the pipeline that is genome-independent.

use std::io::{Read, Write};

/// Channels in the sensory wave, matching `config::iara::brain::SENSORY_WAVE_SIZE`.
pub const WAVE_CHANNELS: usize = 192;
/// f32 per row: the wave plus the three commands the recording being actually issued.
pub const ROW_F32: usize = WAVE_CHANNELS + 3;

pub fn write_row(w: &mut impl Write, wave: &[f32], intent: &[f32; 3]) -> std::io::Result<()> {
    debug_assert_eq!(wave.len(), WAVE_CHANNELS);
    for v in wave.iter().chain(intent.iter()) {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Returns `(waves flattened row-major, one command per row)`.
///
/// A partial row is an error, not a short read: silently dropping the tail is how a
/// screen ends up scoring a candidate against a recording that stopped early.
pub fn read_all(path: &str) -> std::io::Result<(Vec<f32>, Vec<[f32; 3]>)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() % (ROW_F32 * 4) != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{path}: {} bytes is not a whole number of {ROW_F32}-f32 rows",
                bytes.len()
            ),
        ));
    }
    let n_rows = bytes.len() / (ROW_F32 * 4);
    let mut waves = Vec::with_capacity(n_rows * WAVE_CHANNELS);
    let mut intents = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let base = r * ROW_F32 * 4;
        for c in 0..WAVE_CHANNELS {
            let o = base + c * 4;
            waves.push(f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()));
        }
        let mut it = [0.0f32; 3];
        for (k, slot) in it.iter_mut().enumerate() {
            let o = base + (WAVE_CHANNELS + k) * 4;
            *slot = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        }
        intents.push(it);
    }
    Ok((waves, intents))
}
```

Add `pub mod wave_record;` to `src/gpu/mod.rs`.

- [ ] **Step 4: Run it and watch it pass**

Run: `cargo test --release --lib wave_record 2>&1 | grep "test result"`
Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Wire the flag `--gravar-onda <path>`**

In the tick, when the flag is set, after the existing poll, read the wave for being 0 (the same being `--observar` records, so the two files line up tick for tick) and append a row with the intent already read at `context.rs:1330`. Reuse `percept.read_after_submit` exactly as `--auditar-visao` does — do not add a second readback path.

Guard it: recording forces a per-tick readback, so it must be off by default and must be documented as forcing `K = 1` once Phase 1 exists.

- [ ] **Step 6: Record a real champion and check the size**

```bash
df -h /media/llm-data   # 195 f32 x 4 B x 20000 ticks = 15.6 MB per recording
G=/media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin
D=/media/llm-data/scratch/wave; rm -rf $D; mkdir -p $D; cp $G $D/
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 48 --ticks 20000 \
  --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
  --pistas 8005,8200,9001 --seed 77 --gravar-onda $D/wave.f32 --observar $D/obs.f32
ls -la $D/wave.f32
```

Expected: 15,600,000 bytes exactly. Cross-check: the 3 commands in `wave.f32` must equal `obs.f32` columns 298..301 row for row.

- [ ] **Step 7: Commit**

```bash
cargo fmt -p universe
git add -A
git commit -m "feat(gpu_evolve): --gravar-onda records the raw sensory wave

The input side of the open-loop screen. obs.f32 cannot serve: its columns 10..202
are post-KAN and post-CfC, already transformed by the recording genome's own knots
and tau bank, so replaying a different genome from them would hold those stages
fixed and vary only L1/L2. The raw 192-channel wave is the last genome-independent
point in the pipeline.

Round trip is tested including the truncated-file case -- this repository's genome
serialiser silently dropped fields twice, and a screen that scores against a
recording which stopped early would fail the same way.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: The screen — replay recorded waves through the real brain kernel

**Files:**
- Create: `src/bin/screen.rs`
- Modify: `Cargo.toml` (a `[[bin]]` entry if the others are declared explicitly; check first)

**Interfaces:**
- Consumes: `wave_record::read_all` (Task 4), `universe::gpu::brain::BrainGpu`, `universe::gpu::genome_io::load_genome`.
- Produces: binary `screen`, flags `--onda <path> --genomas <dir> [--ticks N] [--limiar 0.26]`; prints one line per genome with mean per-tick error and a verdict.

- [ ] **Step 1: Write the failing test for the scoring rule**

The GPU path is exercised by Step 5's calibration; the arithmetic is unit-tested.

```rust
/// Mean absolute error per tick across the three commands, which is the quantity
/// docs/DESTILABILIDADE.md calibrated: 0.26 compounds in closed loop and the clone
/// stalls, 0.18 does not and it drives.
fn erro_por_tick(pred: &[[f32; 3]], real: &[[f32; 3]]) -> f32 { /* ... */ }

#[cfg(test)]
mod tests {
    use super::erro_por_tick;

    #[test]
    fn a_perfect_clone_scores_zero() {
        let a = vec![[0.5, -0.2, 0.0], [0.1, 0.1, 0.1]];
        assert!(erro_por_tick(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn the_error_is_the_mean_over_ticks_and_channels() {
        let pred = vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let real = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        // one channel off by 1.0, out of 2 ticks x 3 channels = 6 slots
        assert!((erro_por_tick(&pred, &real) - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn mismatched_lengths_score_over_the_common_prefix() {
        let pred = vec![[1.0, 0.0, 0.0]];
        let real = vec![[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]];
        assert!((erro_por_tick(&pred, &real) - 1.0 / 3.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --bin screen 2>&1 | tail -20`
Expected: FAIL — no such binary.

- [ ] **Step 3: Implement the scorer and the driver**

```rust
fn erro_por_tick(pred: &[[f32; 3]], real: &[[f32; 3]]) -> f32 {
    let n = pred.len().min(real.len());
    if n == 0 {
        return f32::INFINITY;
    }
    let mut soma = 0.0f64;
    for i in 0..n {
        for k in 0..3 {
            soma += (pred[i][k] - real[i][k]).abs() as f64;
        }
    }
    (soma / (n as f64 * 3.0)) as f32
}
```

The driver. Reuse the existing brain dispatch — do not reimplement any layer on the CPU:

```rust
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut onda = String::new();
    let mut genomas = String::new();
    let mut ticks_max = usize::MAX;
    let mut limiar = 0.26f32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--onda" => { i += 1; onda = args[i].clone(); }
            "--genomas" => { i += 1; genomas = args[i].clone(); }
            "--ticks" => { i += 1; ticks_max = args[i].parse().unwrap_or(usize::MAX); }
            "--limiar" => { i += 1; limiar = args[i].parse().unwrap_or(0.26); }
            _ => {}
        }
        i += 1;
    }

    let (waves, real) = universe::gpu::wave_record::read_all(&onda)
        .unwrap_or_else(|e| panic!("[ERRO] recording {onda}: {e}"));
    let n_ticks = real.len().min(ticks_max);
    let w = universe::gpu::wave_record::WAVE_CHANNELS;

    let mut arquivos: Vec<std::path::PathBuf> = std::fs::read_dir(&genomas)
        .unwrap_or_else(|e| panic!("[ERRO] {genomas}: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    arquivos.sort();

    println!("recording {n_ticks} ticks · {} candidates · threshold {limiar:.3}",
             arquivos.len());

    for path in &arquivos {
        let nome = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some(g) = universe::gpu::genome_io::load_genome(path.to_str().unwrap()) else {
            println!("  {nome:<34} [ERRO] could not load");
            continue;
        };
        let mut brain = universe::gpu::brain::BrainGpu::new_for_genome(&g)
            .expect("brain device");
        let mut pred: Vec<[f32; 3]> = Vec::with_capacity(n_ticks);
        for t in 0..n_ticks {
            let wave = &waves[t * w..(t + 1) * w];
            pred.push(brain.step_from_wave(wave));
        }
        let e = erro_por_tick(&pred, &real[..n_ticks]);
        let veredito = if e > limiar { "REJECT" } else { "keep" };
        println!("  {nome:<34} error/tick {e:.4}  {veredito}");
    }
}
```

`BrainGpu::new_for_genome` and `step_from_wave` are the two thin wrappers this task adds around the existing dispatch: the first uploads one genome and allocates for `n = 1`, the second writes the wave into the perception output buffer, dispatches the brain, and returns `intent[0..3)`. If the existing `BrainGpu` API already exposes equivalents under other names, use those and delete these — do not add a parallel path.

- [ ] **Step 4: Run it and watch the tests pass**

Run: `cargo test --release --bin screen 2>&1 | grep "test result"`
Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Sanity — the recording genome must score ~0 against its own recording**

```bash
D=/media/llm-data/scratch/wave
mkdir -p $D/g && cp /media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin $D/g/
./target/release/screen --onda $D/wave.f32 --genomas $D/g
```

Expected: error very close to 0. **If it is not, the screen is not reproducing the brain and nothing downstream is trustworthy** — that is the parity check, and it is the whole reason this task reuses the GPU kernel instead of writing a CPU replica.

- [ ] **Step 6: Commit**

```bash
cargo fmt -p universe
git add -A
git commit -m "feat: screen candidate brains open-loop against a recorded champion

Replays a recorded sensory wave through the EXISTING brain kernel for each candidate
genome and scores mean per-tick command error. No perception, no physics, no CPU
reimplementation of any layer -- this repository has already shipped two disjoint
brains once, and a CPU replica would reintroduce exactly that.

The threshold comes from docs/DESTILABILIDADE.md, measured rather than chosen: 0.26
per-tick error compounds in closed loop and the clone stalls; 0.18 does not and it
drives.

Parity check in the commit: the recording genome scores [N] against its own
recording.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: Calibrate the window, and measure the real speedup

**Files:**
- Create: `docs/SCREEN_CALIBRATION.md`
- Modify: `src/bin/screen.rs` (`--ticks` already parsed; no new code expected)

**Interfaces:**
- Consumes: Task 5's binary.
- Produces: a documented recommended window and the measured cost per candidate.

- [ ] **Step 1: Build a labelled set of candidates**

Take the champion and produce degraded copies by attenuating L1/L2 (the mechanism `--retomar-mix` already uses at `main.rs:1938-1946`): mixes 1.00, 0.95, 0.85, 0.70, 0.50. Closed-loop distance for each is the ground truth — one `--gens 1` evaluation per mix, five runs, roughly 4 minutes total.

- [ ] **Step 2: Score the same candidates open-loop at several windows**

```bash
for T in 500 1000 2000 5000 20000; do
  ./target/release/screen --onda $D/wave.f32 --genomas $D/mixes --ticks $T
done
```

- [ ] **Step 3: Find the shortest window that ranks them the same**

Compare the open-loop ranking at each window against the closed-loop distance ranking. Record the shortest window whose ranking agrees, and Spearman correlation at each window.

- [ ] **Step 4: Measure cost per candidate and correct the spec's estimate**

Time the screen for N candidates at the chosen window. The spec claims three orders of magnitude versus ~18 s closed-loop; the estimate came from counting FLOPs, whereas the measured brain kernel is 15% of 33.61 ms/tick at 768 beings. **Write down the measured number even if it is worse than the claim** — the claim is the thing to correct.

- [ ] **Step 5: Write the calibration doc and commit**

Record: the window, the ranking agreement, the measured cost per candidate, and — most importantly — the false-negative rate. A candidate scoring above 0.26 that nevertheless drives well is a false reject, and the funnel's whole value rests on that being rare. If any of the five mixes is a false reject, say so plainly and mark the threshold as unsafe for pruning.

```bash
git add docs/SCREEN_CALIBRATION.md
git commit -m "docs: calibrate the open-loop screen against closed-loop ground truth

[window], [ranking agreement], [measured cost per candidate versus 18 s closed
loop], [false-reject count out of 5].

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Not in this plan

- **Phase 1 (GPU-resident state, K ticks per submit).** Its target kernel is chosen by Task 3's measurement. Writing tasks for it now would mean inventing the target. It gets its own plan once Task 3's table exists.
- **Phase 3 (the world matrix).** Its economics depend on whatever speed Phase 1 produces. Sketched in the spec, planned later.
- **New brain architectures and self-modifying weights.** Own specs, per the spec's section 6.
