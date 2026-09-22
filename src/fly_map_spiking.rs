//! CPU reference for the event-driven spiking fly map.
//!
//! The GPU implementation in `gpu/fly_map/spiking.rs` is deliberately compared with this
//! small, deterministic model.  It uses the v2 presynaptic CSR, a fixed delay ring and the
//! same current-based LIF equations as `neuromod-lab/src/flybrain/dynamics.py`.

use super::{EyeFrame, FlyMap, FlySelection, NONE};
use std::io::{self, Read, Seek, SeekFrom, Write};

/// Magic for the optional event trace consumed by `fly_map_view`.
pub const TRACE_MAGIC: &[u8; 8] = b"FSPKTR01";

/// One sampled being from the GPU-resident spiking state.
#[derive(Clone, Debug, PartialEq)]
pub struct SpikingTraceFrame {
    pub step: u64,
    pub commands: [f32; super::COMMANDS],
    pub v: Vec<f32>,
    pub spike_counts: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpikingTrace {
    pub n: usize,
    pub being: usize,
    pub frames: Vec<SpikingTraceFrame>,
}

/// Header for the opt-in binary trace. The selected being is recorded so a viewer cannot
/// accidentally label a trace as being 0 after the caller asked for another one.
pub fn write_trace_header(w: &mut impl Write, n: usize, being: usize) -> io::Result<()> {
    let n = u32::try_from(n).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "spiking trace neuron count does not fit in u32",
        )
    })?;
    let being = u32::try_from(being).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "spiking trace being index does not fit in u32",
        )
    })?;
    if n == 0 || being >= n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spiking trace requires being < neuron count and a non-empty map",
        ));
    }
    w.write_all(TRACE_MAGIC)?;
    w.write_all(&n.to_le_bytes())?;
    w.write_all(&being.to_le_bytes())?;
    w.write_all(&1u32.to_le_bytes())
}

/// Appends one frame. Arrays are length-prefixed in the header and fixed-size afterwards,
/// which keeps the hot path allocation-free and lets the viewer stream records to EOF.
pub fn write_trace_frame(w: &mut impl Write, frame: &SpikingTraceFrame) -> io::Result<()> {
    if frame.v.len() != frame.spike_counts.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "spiking trace voltage/count arrays have different lengths",
        ));
    }
    w.write_all(&frame.step.to_le_bytes())?;
    for value in frame.commands {
        w.write_all(&value.to_le_bytes())?;
    }
    for value in &frame.v {
        w.write_all(&value.to_le_bytes())?;
    }
    for value in &frame.spike_counts {
        w.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// Reads the complete optional trace. The viewer uses this bounded diagnostic format, not
/// the hot simulation loop; malformed or truncated records are rejected with a useful
/// error instead of being rendered as shifted neuron activations.
pub fn read_trace(path: &str) -> io::Result<SpikingTrace> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 20 || &bytes[..8] != TRACE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: not an FSPKTR01 trace"),
        ));
    }
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let being = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let version = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    if n == 0 || version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: unsupported trace header n={n}, version={version}"),
        ));
    }
    let record_bytes = 8 + super::COMMANDS * 4 + n * 4 + n * 4;
    if (bytes.len() - 20) % record_bytes != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: truncated record (expected {record_bytes} bytes per frame)"),
        ));
    }
    let mut frames = Vec::with_capacity((bytes.len() - 20) / record_bytes);
    let mut at = 20;
    while at < bytes.len() {
        let step = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        at += 8;
        let mut commands = [0.0; super::COMMANDS];
        for command in &mut commands {
            *command = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            at += 4;
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
            at += 4;
        }
        let mut spike_counts = Vec::with_capacity(n);
        for _ in 0..n {
            spike_counts.push(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
            at += 4;
        }
        frames.push(SpikingTraceFrame {
            step,
            commands,
            v,
            spike_counts,
        });
    }
    if being >= n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: trace being index {being} is outside map of {n} neurons"),
        ));
    }
    Ok(SpikingTrace { n, being, frames })
}

/// Reads at most `max_frames` frames without loading the complete trace into memory.
///
/// A full-map voltage/count frame is about 1.1 MiB (`n × (f32 + u32)`), so opening a
/// long trace by first decoding every frame can turn a diagnostic viewer into a second
/// memory consumer.  The viewer uses this bounded seek-based reader: it keeps the first
/// and last samples and spaces the remaining samples across the file.  The on-disk
/// format stays unchanged and `read_trace` remains available when callers really need
/// every frame.
pub fn read_trace_sampled(path: &str, max_frames: usize) -> io::Result<SpikingTrace> {
    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 20];
    file.read_exact(&mut header)?;
    if &header[..8] != TRACE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: not an FSPKTR01 trace"),
        ));
    }
    let n = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let being = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let version = u32::from_le_bytes(header[16..20].try_into().unwrap());
    if n == 0 || being >= n || version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: unsupported trace header n={n}, being={being}, version={version}"),
        ));
    }
    let record_bytes = 8usize
        .checked_add(super::COMMANDS * 4)
        .and_then(|v| v.checked_add(n.checked_mul(8)?))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "trace record size overflow"))?;
    let file_bytes = file.metadata()?.len();
    let payload_bytes = file_bytes
        .checked_sub(20)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "trace header is truncated"))?;
    if payload_bytes % record_bytes as u64 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: truncated record (expected {record_bytes} bytes per frame)"),
        ));
    }
    let total = (payload_bytes / record_bytes as u64) as usize;
    let keep = max_frames.max(1).min(total);
    let mut frames = Vec::with_capacity(keep);
    if total > 0 {
        let mut record = vec![0u8; record_bytes];
        for sample in 0..keep {
            // Integer interpolation gives deterministic first/last coverage and avoids
            // duplicate seeks when `keep < total`.
            let index = if keep == 1 {
                0
            } else {
                sample * (total - 1) / (keep - 1)
            };
            let offset =
                20u64
                    .checked_add((index as u64).checked_mul(record_bytes as u64).ok_or_else(
                        || io::Error::new(io::ErrorKind::InvalidData, "trace seek offset overflow"),
                    )?)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "trace seek offset overflow")
                    })?;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut record)?;
            let mut at = 0;
            let step = u64::from_le_bytes(record[at..at + 8].try_into().unwrap());
            at += 8;
            let mut commands = [0.0; super::COMMANDS];
            for command in &mut commands {
                *command = f32::from_le_bytes(record[at..at + 4].try_into().unwrap());
                at += 4;
            }
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(f32::from_le_bytes(record[at..at + 4].try_into().unwrap()));
                at += 4;
            }
            let mut spike_counts = Vec::with_capacity(n);
            for _ in 0..n {
                spike_counts.push(u32::from_le_bytes(record[at..at + 4].try_into().unwrap()));
                at += 4;
            }
            frames.push(SpikingTraceFrame {
                step,
                commands,
                v,
                spike_counts,
            });
        }
    }
    Ok(SpikingTrace { n, being, frames })
}

/// Input mode for the eye binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeMode {
    /// Dark pixels produce the high-rate histaminergic drive used by R1-6.
    Photoreceptor,
    /// Absolute frame-to-frame luminance change.  This is motion energy, not direction
    /// selectivity; a direction-aware T4/T5 map needs an additional polarity/channel table.
    Motion,
}

/// Numerical parameters shared by the CPU reference and WGSL.
#[derive(Clone, Copy, Debug)]
pub struct SpikingParams {
    pub dt_ms: f32,
    pub v0: f32,
    pub v_rst: f32,
    pub v_th: f32,
    pub tau_m_ms: f32,
    pub tau_syn_ms: f32,
    pub refractory_ms: f32,
    pub delay_ms: f32,
    pub w_syn: f32,
    pub poisson_factor: f32,
    pub background_hz: f32,
    pub read_tau_ms: f32,
}

impl Default for SpikingParams {
    fn default() -> Self {
        Self {
            dt_ms: 0.1,
            v0: -52.0,
            v_rst: -52.0,
            v_th: -45.0,
            tau_m_ms: 20.0,
            tau_syn_ms: 5.0,
            refractory_ms: 2.2,
            delay_ms: 1.8,
            w_syn: 0.275,
            poisson_factor: 250.0,
            background_hz: 250.0,
            read_tau_ms: 50.0,
        }
    }
}

impl SpikingParams {
    pub fn delay_steps(self) -> usize {
        (self.delay_ms / self.dt_ms).round().max(1.0) as usize
    }

    pub fn refractory_steps(self) -> u32 {
        (self.refractory_ms / self.dt_ms).round().max(1.0) as u32
    }

    pub fn decay_v(self) -> f32 {
        (-self.dt_ms / self.tau_m_ms).exp()
    }

    pub fn decay_syn(self) -> f32 {
        (-self.dt_ms / self.tau_syn_ms).exp()
    }

    pub fn decay_read(self, tick_dt_ms: f32) -> f32 {
        (-tick_dt_ms / self.read_tau_ms).exp()
    }
}

/// Persistent state, with all neuron rows using a fixed `capacity` stride.
pub struct SpikingState {
    pub v: Vec<f32>,
    pub i_syn: Vec<f32>,
    pub refractory: Vec<u32>,
    ring: Vec<bool>,
    spike_counts: Vec<u32>,
    read_count: Vec<f32>,
    pub step: u64,
    pub capacity: usize,
    pub delay_steps: usize,
}

impl SpikingState {
    pub fn new(n: usize, capacity: usize, params: SpikingParams) -> Self {
        let cap = capacity.max(1);
        let d = params.delay_steps();
        let len = n * cap;
        Self {
            v: vec![params.v0; len],
            i_syn: vec![0.0; len],
            refractory: vec![0; len],
            ring: vec![false; d * len],
            spike_counts: vec![0; len],
            read_count: vec![0.0; cap * super::COMMANDS],
            step: 0,
            capacity: cap,
            delay_steps: d,
        }
    }

    pub fn reset(&mut self, params: SpikingParams) {
        self.v.fill(params.v0);
        self.i_syn.fill(0.0);
        self.refractory.fill(0);
        self.ring.fill(false);
        self.spike_counts.fill(0);
        self.read_count.fill(0.0);
        self.step = 0;
    }

    fn idx(&self, neuron: usize, being: usize) -> usize {
        neuron * self.capacity + being
    }

    fn ring_idx(&self, slot: usize, neuron: usize, being: usize, n: usize) -> usize {
        (slot * n + neuron) * self.capacity + being
    }

    /// One LIF substep. Returns a flat `(neuron, being)` spike bitmap.
    pub fn step(
        &mut self,
        map: &FlyMap,
        sel: &FlySelection,
        n_beings: usize,
        params: SpikingParams,
        eye: Option<&EyeFrame<'_>>,
        eye_mode: EyeMode,
        previous_retina: Option<&[u32]>,
        seed: u64,
    ) -> Vec<bool> {
        let n = map.n();
        let b = n_beings.min(self.capacity).min(sel.n_beings);
        assert_eq!(self.delay_steps, params.delay_steps());
        let slot = (self.step as usize) % self.delay_steps;

        // Propagate delivered spikes before integration, exactly as LifBrain.step().
        if let Some(pre) = &map.pre_csr {
            for source in 0..n {
                let type_idx = map.type_of[source];
                for being in 0..b {
                    let delivered = self.ring[self.ring_idx(slot, source, being, n)];
                    if !delivered
                        || sel.type_mask[being * sel.layout.n_types + type_idx as usize] < 0.5
                    {
                        continue;
                    }
                    for edge in pre.indptr[source] as usize..pre.indptr[source + 1] as usize {
                        let post = pre.indices[edge] as usize;
                        let post_idx = self.idx(post, being);
                        self.i_syn[post_idx] +=
                            params.w_syn * pre.weights[edge] * sel.scale(being).max(0.0);
                    }
                }
            }
        }

        let dec_v = params.decay_v();
        let dec_syn = params.decay_syn();
        let mut out = vec![false; n * self.capacity];
        for neuron in 0..n {
            let pixel = map
                .pixel_of_neuron
                .as_ref()
                .map(|p| p[neuron])
                .unwrap_or(NONE);
            for being in 0..b {
                let idx = self.idx(neuron, being);
                let free = self.refractory[idx] == 0;
                let syn = self.i_syn[idx];
                let mut spiked = false;
                if free {
                    self.v[idx] =
                        params.v0 + (self.v[idx] - params.v0) * dec_v + syn * (1.0 - dec_v);
                    self.i_syn[idx] = syn * dec_syn;
                    if let Some(frame) = eye {
                        if pixel != NONE {
                            let rate_factor = eye_rate_factor(
                                frame,
                                previous_retina,
                                being,
                                pixel as usize,
                                eye_mode,
                            );
                            let rate =
                                (params.background_hz * sel.gain(being).max(0.0) * rate_factor)
                                    .max(0.0);
                            let p = poisson_probability(rate, params.dt_ms);
                            if hash_uniform(neuron as u32, being as u32, self.step as u32, seed) < p
                            {
                                self.v[idx] += params.w_syn * params.poisson_factor;
                            }
                        }
                    }
                    spiked = self.v[idx] > params.v_th;
                } else {
                    // During the absolute refractory interval no new event is
                    // integrated or emitted. The accumulated synaptic current is kept
                    // and consumed on the first free substep, preserving this model's
                    // current-based state contract.
                    self.refractory[idx] -= 1;
                }

                if spiked {
                    self.v[idx] = params.v_rst;
                    self.i_syn[idx] = 0.0;
                    if !is_no_refractory(map, neuron) {
                        self.refractory[idx] = params.refractory_steps();
                    }
                    self.spike_counts[idx] = 1;
                }
                out[idx] = spiked;
            }
        }
        // The current slot is overwritten after it has been consumed by propagation.
        for neuron in 0..n {
            for being in 0..b {
                let idx = self.ring_idx(slot, neuron, being, n);
                self.ring[idx] = out[self.idx(neuron, being)];
            }
        }
        self.step += 1;
        out
    }

    /// Applies the K3 readout to the spikes emitted during the current Universe tick.
    /// Callers may execute several substeps before this method; counts are accumulated
    /// across those substeps and consumed here, matching the GPU tick boundary.
    pub fn readout(
        &mut self,
        map: &FlyMap,
        sel: &FlySelection,
        n_beings: usize,
        params: SpikingParams,
        tick_dt_ms: f32,
    ) -> Vec<[f32; super::COMMANDS]> {
        let b = n_beings.min(self.capacity).min(sel.n_beings);
        let decay = params.decay_read(tick_dt_ms);
        let keep = 1.0 - decay;
        let mut out = vec![[0.0; super::COMMANDS]; b];
        for being in 0..b {
            for command in 0..super::COMMANDS {
                let plus = sel.out_pair[being * super::OUT_SLOTS + command * 2];
                let minus = sel.out_pair[being * super::OUT_SLOTS + command * 2 + 1];
                let drive = group_spike_mean(map, &self.spike_counts, self.capacity, plus, being)
                    - group_spike_mean(map, &self.spike_counts, self.capacity, minus, being);
                let idx = being * super::COMMANDS + command;
                self.read_count[idx] = self.read_count[idx] * decay + keep * drive;
                out[being][command] = (self.read_count[idx]).tanh();
            }
        }
        self.spike_counts.fill(0);
        out
    }

    /// Starts a new readout window without consuming it. This is useful for diagnostics
    /// that want to inspect state between ticks; normal callers should let [`readout`]
    /// clear the window after K3 has consumed it.
    pub fn clear_spike_counts(&mut self) {
        self.spike_counts.fill(0);
    }
}

fn is_no_refractory(map: &FlyMap, neuron: usize) -> bool {
    map.no_refractory
        .as_ref()
        .map(|words| (words[neuron / 32] & (1u32 << (neuron & 31))) != 0)
        .unwrap_or(false)
}

fn group_spike_mean(
    map: &FlyMap,
    spikes: &[u32],
    capacity: usize,
    group: u32,
    being: usize,
) -> f32 {
    if group == NONE || group as usize >= map.k_out() {
        return 0.0;
    }
    let start = map.out_indptr[group as usize] as usize;
    let end = map.out_indptr[group as usize + 1] as usize;
    if start == end {
        return 0.0;
    }
    let sum: u32 = map.out_members[start..end]
        .iter()
        .map(|&neuron| spikes[neuron as usize * capacity + being])
        .sum();
    sum as f32 / (end - start) as f32
}

fn eye_rate_factor(
    frame: &EyeFrame<'_>,
    previous: Option<&[u32]>,
    being: usize,
    pixel: usize,
    mode: EyeMode,
) -> f32 {
    let current = frame.retina[being * frame.retina_stride + pixel];
    let current = packed_brightness(current);
    match mode {
        EyeMode::Photoreceptor => 1.0 - current,
        EyeMode::Motion => {
            let previous = previous
                .and_then(|p| p.get(being * frame.retina_stride + pixel).copied())
                .map(packed_brightness)
                .unwrap_or(current);
            (current - previous).abs().clamp(0.0, 1.0)
        }
    }
}

fn packed_brightness(packed: u32) -> f32 {
    let dist_m = (packed >> 8) as f32 / 1000.0;
    let material = (packed & 0xFF) as u8;
    let raw = crate::gpu::legacy_encoding(crate::gpu::Pixel { dist_m, material });
    ((raw as f32 + 128.0) / 255.0).clamp(0.0, 1.0)
}

/// Exact Bernoulli probability for at least one event in a Poisson interval.
pub fn poisson_probability(rate_hz: f32, dt_ms: f32) -> f32 {
    if !rate_hz.is_finite() || !dt_ms.is_finite() || rate_hz <= 0.0 || dt_ms <= 0.0 {
        return 0.0;
    }
    (1.0 - (-rate_hz * dt_ms * 1e-3).exp()).clamp(0.0, 1.0)
}

fn hash_uniform(neuron: u32, being: u32, step: u32, seed: u64) -> f32 {
    let mut x = neuron
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(being.wrapping_mul(0x85EB_CA6B))
        .wrapping_add(step.wrapping_mul(0xC2B2_AE35))
        .wrapping_add(seed as u32)
        .wrapping_add((seed >> 32) as u32);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x >> 8) as f32 / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fly_map::{pack_selection, FlyLayout};

    #[test]
    fn poisson_probability_is_exact_and_bounded() {
        let p = poisson_probability(250.0, 0.1);
        assert!((p - (1.0 - (-0.025f32).exp())).abs() < 1e-7);
        assert_eq!(poisson_probability(-1.0, 0.1), 0.0);
        assert_eq!(poisson_probability(f32::INFINITY, 0.1), 0.0);
        assert!(poisson_probability(1e9, 1000.0) <= 1.0);
    }

    #[test]
    fn delay_ring_delivers_at_exact_step() {
        let mut map = FlyMap::synthetic();
        map.pre_csr = Some(crate::fly_map::PresynapticCsr {
            indptr: vec![0, 1, 1, 1, 1, 1, 1],
            indices: vec![1],
            weights: vec![1.0],
        });
        let layout = FlyLayout {
            k_in: map.k_in,
            k_out: map.k_out(),
            n_types: map.n_types,
        };
        let mut block = vec![-1.0; layout.block_len()];
        block[crate::fly_map::OFF_GAIN] = 1.0;
        block[crate::fly_map::OFF_SCALE] = 1.0;
        for t in 0..layout.n_types {
            block[crate::fly_map::OFF_TYPE_MASK + t] = 1.0;
        }
        let sel = pack_selection(&[block.as_slice()], layout, 1);
        let params = SpikingParams::default();
        let mut state = SpikingState::new(map.n(), 1, params);
        // A direct impulse is injected in the first source's state.  The synthetic map
        // has source 0 -> postsynaptic neuron 1, so only step D may deliver it.
        state.v[0] = params.v_th + 1.0;
        for _ in 0..params.delay_steps() {
            state.step(&map, &sel, 1, params, None, EyeMode::Photoreceptor, None, 0);
        }
        assert_eq!(state.i_syn[1], 0.0, "the impulse must not arrive before D");
        state.step(&map, &sel, 1, params, None, EyeMode::Photoreceptor, None, 0);
        assert!(state.i_syn[1] > 0.0, "the impulse must arrive at exactly D");
    }

    #[test]
    fn refractory_blocks_a_second_spike() {
        let map = FlyMap::synthetic();
        let layout = map.fly_layout();
        let mut block = vec![-1.0; layout.block_len()];
        block[crate::fly_map::OFF_GAIN] = 1.0;
        block[crate::fly_map::OFF_SCALE] = 1.0;
        for t in 0..layout.n_types {
            block[crate::fly_map::OFF_TYPE_MASK + t] = 1.0;
        }
        let sel = pack_selection(&[block.as_slice()], layout, 1);
        let params = SpikingParams::default();
        let mut state = SpikingState::new(map.n(), 1, params);
        state.v[0] = params.v_th + 1.0;
        let first = state.step(&map, &sel, 1, params, None, EyeMode::Photoreceptor, None, 0);
        assert!(first[0]);
        state.v[0] = params.v_th + 1.0;
        let second = state.step(&map, &sel, 1, params, None, EyeMode::Photoreceptor, None, 0);
        assert!(!second[0], "refractory neuron emitted a second event");
    }

    #[test]
    fn trace_round_trip_preserves_header_and_frame() {
        let path = std::env::temp_dir().join(format!(
            "universe-fspktrace-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let frame = SpikingTraceFrame {
            step: 42,
            commands: [0.25, -0.5, 1.0],
            v: vec![-52.0, -44.75],
            spike_counts: vec![0, 3],
        };
        {
            let mut file = std::fs::File::create(&path).expect("create trace");
            write_trace_header(&mut file, 2, 1).expect("write header");
            write_trace_frame(&mut file, &frame).expect("write frame");
        }
        let trace = read_trace(path.to_str().expect("utf8 temp path")).expect("read trace");
        assert_eq!(trace.n, 2);
        assert_eq!(trace.being, 1);
        assert_eq!(trace.frames, vec![frame]);
        std::fs::remove_file(path).expect("remove trace");
    }

    #[test]
    fn sampled_trace_keeps_endpoints_without_decoding_every_frame() {
        let path = std::env::temp_dir().join(format!(
            "universe-fspktrace-sampled-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        {
            let mut file = std::fs::File::create(&path).expect("create trace");
            write_trace_header(&mut file, 1, 0).expect("write header");
            for step in 0..5u64 {
                write_trace_frame(
                    &mut file,
                    &SpikingTraceFrame {
                        step,
                        commands: [step as f32, 0.0, 0.0],
                        v: vec![-52.0 + step as f32],
                        spike_counts: vec![step as u32],
                    },
                )
                .expect("write frame");
            }
        }
        let trace = read_trace_sampled(path.to_str().expect("utf8 temp path"), 3)
            .expect("read sampled trace");
        assert_eq!(trace.frames.len(), 3);
        assert_eq!(
            trace
                .frames
                .iter()
                .map(|frame| frame.step)
                .collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
        std::fs::remove_file(path).expect("remove trace");
    }
}
