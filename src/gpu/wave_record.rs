//! Raw sensory-wave recording: the input side of the open-loop screen.
//!
//! `obs.f32` is NOT enough for this. Its columns 10..202 are post-KAN and post-CfC —
//! already transformed by the recording genome's own KAN knots and tau bank. Replaying
//! a DIFFERENT genome from them would hold those stages fixed and only vary L1/L2. The
//! raw 192-channel wave is the last point in the pipeline that is genome-independent.
//!
//! # Config header (added in the Task 5 review round — `UWAV` v1)
//!
//! A recording is not fully described by the wave and the commands: the brain's
//! forward pass and the recorded intent both depend on ten runtime flags
// check-language: off -- quoting gpu_evolve's existing (Portuguese) CLI flag names
// verbatim, for anyone grepping this doc against `main.rs`'s actual flag table.
//! (`--out-beta`, `--ganho-saida`, `--norm-l2-por-desvio`/`--sem-norm-l2-por-desvio`,
//! `--ruido-sensorial`, `--ruido-dropout`, `--sem-memoria-neuronal`,
//! `--sem-periodo-neuronal`, `--sem-limiar-neuronal`, `--sem-ativacao-heterogenea`,
//! `--neuromod-regioes`) plus `dt`, `--brain-skip` and whether a ghost pilot
//! (`--piloto-forja`/`--piloto-formula`) was overwriting the recorded commands.
// check-language: on
//! None of that lived in the file before this header: the screen had to guess `out_beta`
//! (`--out-beta` on the screen's own CLI), and guessing is exactly what produced a
//! confident-looking but wrong parity score of 0.4772 against a recording made under
//! `out_beta=0.0`. A legacy (headerless) file is now rejected outright rather than
//! defaulted — defaulting is the bug this header exists to remove.

use std::io::{Read, Write};

use crate::config::iara::brain::SENSORY_WAVE_SIZE;

/// Channels in the sensory wave, matching `config::iara::brain::SENSORY_WAVE_SIZE`.
pub const WAVE_CHANNELS: usize = 192;

// One source of truth per constant (`docs/STYLE.md` §4): `WAVE_CHANNELS` is a literal
// because a recorded `.f32` file's row width cannot follow `SENSORY_WAVE_SIZE` if that
// constant ever changes underneath an existing recording — but the two must never
// silently drift apart, so the agreement is pinned here instead of trusted by comment.
const _: () = assert!(
    WAVE_CHANNELS == SENSORY_WAVE_SIZE,
    "WAVE_CHANNELS must track config::iara::brain::SENSORY_WAVE_SIZE"
);

/// f32 per row: the wave plus the three commands the recording being actually issued.
pub const ROW_F32: usize = WAVE_CHANNELS + 3;

/// Words in the "4 i8 channels per u32" packing `perception.wgsl`'s `buf_output` and
/// `slow_gather.wgsl`'s `buf_wave` binding share. Pinned against
/// `gpu::perception::WAVE_U32S` the same way `WAVE_CHANNELS` is pinned above — this
/// project has already scrambled the slow brain's input once from an unguarded copy of
/// this exact constant (`slow_gather.wgsl`'s own comment: "was 56u (223-channel wave)").
pub const WAVE_U32S: usize = WAVE_CHANNELS.div_ceil(4); // 48

const _: () = assert!(
    WAVE_U32S == crate::gpu::perception::WAVE_U32S,
    "wave_record::WAVE_U32S must track gpu::perception::WAVE_U32S"
);

/// Packs one recorded wave (already exact i8-per-channel values, widened to f32 by
/// `read_all`) into the packed `WAVE_U32S`-word layout the slow brain's
/// `SlowBrainGpu::dispatch_gather` reads from, replicated `n` times.
///
/// The screen scores many candidate genomes against the SAME recorded tick (Task 5
/// review, C1/C2): the slow brain — needed to reproduce the M4 region gates — has to
/// see `n` identical copies of this one wave, one per being slot in the batch.
pub fn pack_wave_u32(wave: &[f32], n: usize) -> Vec<u32> {
    debug_assert_eq!(wave.len(), WAVE_CHANNELS);
    let mut block = [0u32; WAVE_U32S];
    for (ch, &v) in wave.iter().enumerate() {
        // Exact: `v` is an i8 channel value widened to f32 earlier in the pipeline
        // (`read_after_submit`/`wave_record::read_all`), never rounded.
        let byte = (v as i8) as u8;
        block[ch / 4] |= (byte as u32) << ((ch % 4) * 8);
    }
    let mut out = Vec::with_capacity(n * WAVE_U32S);
    for _ in 0..n {
        out.extend_from_slice(&block);
    }
    out
}

/// Magic bytes identifying a `UWAV` v1 recording (config header + rows). Same
/// magic-sniff pattern as `genome_io::MAGIC` (`UGEN`): 4 ASCII bytes at offset 0.
pub const MAGIC: &[u8; 4] = b"UWAV";

/// Current format version. Bump it whenever `WaveConfig`'s layout changes.
pub const FORMAT_VERSION: u32 = 1;

/// The runtime configuration a wave recording was made under.
///
/// Every field here is a `gpu_evolve` flag (or the tick length it implies) that
/// changes what the brain kernel computes or what ends up in the recorded intent —
/// see the module doc for the list and why each one matters. `bool`s are stored as
/// `u32` (0/1) because `bytemuck::Pod` requires a fixed-width, no-niche
/// representation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaveConfig {
    /// Seconds per tick the recording ran at (`gpu_evolve` uses 1/64 today). The brain
    /// kernel's CfC decay is `exp(-d*dt)` (`brain.wgsl:684`) with `dt` read directly
    /// from this value, computed once per DISPATCH — i.e. per recorded row, never
    /// scaled by `brain_skip` (skipped ticks never dispatch the kernel at all, so its
    /// internal notion of elapsed time only ever advances by `dt` per row). Replaying
    /// rows in order with this same `dt` per row is therefore correct independently of
    /// `brain_skip`'s value; `brain_skip` is still recorded below, for provenance and
    /// so a reader does not have to take that on faith.
    pub dt: f32,
    /// `--out-beta`. `BrainGpu::out_beta` — output-fatigue shrinkage constant.
    pub out_beta: f32,
    /// `--ganho-saida`. `BrainGpu::out_gain`.
    pub out_gain: f32,
    /// `--norm-l2-por-desvio` (1) / `--sem-norm-l2-por-desvio` (0). `BrainGpu::norm_l2_by_std`.  // check-language: allow -- quoting the flag name verbatim
    pub norm_l2_by_std: u32,
    /// `--ruido-sensorial`. `BrainGpu::noise_sigma`, in i8 channel units. Non-zero
    /// makes a recording irreproducible at ANY setting (see `dropout_prob`) because the
    /// recorder captures the pre-noise wave (`percept.buf_output()`) while the commands
    /// were produced from the post-noise wave inside the shader.
    pub noise_sigma: f32,
    /// `--ruido-dropout`. `BrainGpu::dropout_prob`. Same irreproducibility as
    /// `noise_sigma`, same reason.
    pub dropout_prob: f32,
    /// `--sem-memoria-neuronal` (1 = mechanism forced OFF). `BrainGpu::mem_off`.  // check-language: allow -- quoting the flag name verbatim
    pub mem_off: u32,
    /// `--sem-periodo-neuronal` (1 = forced OFF). `BrainGpu::period_off`.  // check-language: allow -- quoting the flag name verbatim
    pub period_off: u32,
    /// `--sem-limiar-neuronal` (1 = forced OFF). `BrainGpu::threshold_off`.  // check-language: allow -- quoting the flag name verbatim
    pub threshold_off: u32,
    /// `--sem-ativacao-heterogenea` (1 = forced OFF). `BrainGpu::activation_off`.  // check-language: allow -- quoting the flag name verbatim
    pub activation_off: u32,
    /// `--neuromod-regioes` (1 = M4 region gates active). When set, `buf_gates` is not
    /// neutral — a screen that leaves it at 1.0 is scoring an approximation, not the
    /// literal recording; see `docs/` Task 5 reports for the measured effect.
    pub neuromod_regioes: u32,
    /// `--brain-skip` (ticks between brain firings; 1 = fires every tick, the default).
    pub brain_skip: u32,
    /// 1 if `--piloto-forja` or `--piloto-formula` was active. Either one overwrites
    /// `intents[0]` (`context.rs:1693,1716`) BEFORE the recorder reads it
    /// (`context.rs:2022`) — the recorded "champion commands" would be a ghost
    /// pilot's output, not the brain's. A recording made this way cannot be screened
    /// against any brain at all and must be rejected, not just flagged.
    pub pilot_active: u32,
}

/// Ticks recorded so far. Reserved for the writer side (`--gravar-onda`) to report a
/// row count without re-deriving it from bytes written; not yet read by anything.
pub struct WaveRecord {
    pub ticks: usize,
}

/// Writes the `UWAV` v1 header: magic, version, then `cfg` verbatim. Must be called
/// exactly once, before any `write_row` call, on a freshly created file.
pub fn write_header(w: &mut impl Write, cfg: &WaveConfig) -> std::io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    w.write_all(bytemuck::bytes_of(cfg))?;
    Ok(())
}

/// Appends one row: `wave` (`WAVE_CHANNELS` f32) followed by `intent` (3 f32), all
/// little-endian. Panics via `debug_assert_eq!` in debug builds if `wave` is the wrong
/// length — a silently short or long wave would desync every row after it.
pub fn write_row(w: &mut impl Write, wave: &[f32], intent: &[f32; 3]) -> std::io::Result<()> {
    debug_assert_eq!(wave.len(), WAVE_CHANNELS);
    for v in wave.iter().chain(intent.iter()) {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn invalid_data(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

/// Reads a `UWAV` v1 recording: the config header, the wave rows flattened row-major,
/// and one command per row.
///
/// Rejects, rather than defaults:
/// - **no `UWAV` magic** (a legacy, headerless recording, or any other file): the
///   runtime parameters are unknown and a wrong guess is exactly the bug this header
///   exists to remove. Re-record with the current `gpu_evolve`.
/// - **`noise_sigma > 0` or `dropout_prob > 0`**: irreproducible at any setting — see
///   `WaveConfig::noise_sigma`.
/// - **`pilot_active != 0`**: the recorded commands are a ghost pilot's output, not
///   the brain's — see `WaveConfig::pilot_active`.
/// - **a partial row**: silently dropping the tail is how a screen ends up scoring a
///   candidate against a recording that stopped early.
pub fn read_all(path: &str) -> std::io::Result<(Vec<f32>, Vec<[f32; 3]>, WaveConfig)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;

    let header_len = 8 + std::mem::size_of::<WaveConfig>();
    if bytes.len() < header_len || &bytes[0..4] != MAGIC {
        return Err(invalid_data(format!(
            "{path}: no UWAV header found. This is a legacy recording (or not a wave \
             recording at all) — its runtime brain parameters (out_beta, noise, \
             neuromod-regioes, brain_skip, …) are unknown and CANNOT be safely assumed. \
             Re-record it with the current gpu_evolve (--gravar-onda writes the header \
             now)."
        )));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    // `version == 0` used to fall through the old `version > FORMAT_VERSION` check and
    // be read as v1 — a 0 here is not an old format to fall back to (the magic-sniff
    // check above already requires the `UWAV` header, so nothing that predates
    // versioning gets this far); it is a corrupt or hand-edited file, and must be
    // rejected the same as a version newer than this binary understands.
    if version == 0 || version > FORMAT_VERSION {
        return Err(invalid_data(format!(
            "{path}: UWAV version {version} is not a supported version (this binary \
             understands {FORMAT_VERSION}). Rebuild the screen if this file is newer, \
             or re-record it if it is corrupt."
        )));
    }
    let config: WaveConfig = *bytemuck::from_bytes(&bytes[8..header_len]);

    if config.noise_sigma > 0.0 || config.dropout_prob > 0.0 {
        return Err(invalid_data(format!(
            "{path}: recorded with sensor noise (sigma={}, dropout={}) — irreproducible \
             at any setting. The recorder captures the PRE-noise wave while the commands \
             were produced from the POST-noise wave inside the shader; there is no config \
             that makes replaying the pre-noise wave match. Re-record without \
             --ruido-sensorial/--ruido-dropout.",
            config.noise_sigma, config.dropout_prob
        )));
    }
    if config.pilot_active != 0 {
        return Err(invalid_data(format!(
            "{path}: recorded with a ghost pilot active (--piloto-forja or \
             --piloto-formula) — the recorded commands are the pilot's output, not the \
             brain's. Re-record without either flag."
        )));
    }

    let body = &bytes[header_len..];
    if body.len() % (ROW_F32 * 4) != 0 {
        return Err(invalid_data(format!(
            "{path}: {} body bytes is not a whole number of {ROW_F32}-f32 rows",
            body.len()
        )));
    }
    let n_rows = body.len() / (ROW_F32 * 4);
    let mut waves = Vec::with_capacity(n_rows * WAVE_CHANNELS);
    let mut intents = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let base = r * ROW_F32 * 4;
        for c in 0..WAVE_CHANNELS {
            let o = base + c * 4;
            waves.push(f32::from_le_bytes(body[o..o + 4].try_into().unwrap()));
        }
        let mut it = [0.0f32; 3];
        for (k, slot) in it.iter_mut().enumerate() {
            let o = base + (WAVE_CHANNELS + k) * 4;
            *slot = f32::from_le_bytes(body[o..o + 4].try_into().unwrap());
        }
        intents.push(it);
    }
    Ok((waves, intents, config))
}

#[cfg(test)]
mod tests {
    use super::{
        MAGIC, WAVE_CHANNELS, WAVE_U32S, WaveConfig, pack_wave_u32, read_all, write_header,
        write_row,
    };
    use std::io::Write as _;

    fn sample_config() -> WaveConfig {
        WaveConfig {
            dt: 1.0 / 64.0,
            out_beta: 0.0,
            out_gain: 2.0,
            norm_l2_by_std: 1,
            noise_sigma: 0.0,
            dropout_prob: 0.0,
            mem_off: 0,
            period_off: 0,
            threshold_off: 0,
            activation_off: 0,
            neuromod_regioes: 1,
            brain_skip: 1,
            pilot_active: 0,
        }
    }

    /// Owns a per-test scratch directory and removes it on drop, including on a
    /// panicking assertion — a manual `remove_dir_all` at the end of the test body
    /// never runs in that case, which is how this file's tests leaked 60 `/tmp/wave_*`
    /// directories across roughly a dozen `cargo test` runs before this guard existed.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("wave_{tag}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        fn file(&self, name: &str) -> std::path::PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The format has silently dropped fields before (the slow brain's 1091 weights,
    /// the 96-entry tau bank, in `genome_io`) and now carries a whole config struct —
    /// round-tripped before anything depends on it.
    #[test]
    fn a_recording_survives_the_disk_with_its_header() {
        let dir = TempDir::new("rt");
        let path = dir.file("w.f32");
        let cfg = sample_config();
        let wave_a: Vec<f32> = (0..WAVE_CHANNELS).map(|i| i as f32 * 0.5).collect();
        let wave_b: Vec<f32> = (0..WAVE_CHANNELS).map(|i| -(i as f32)).collect();
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            write_header(&mut f, &cfg).unwrap();
            write_row(&mut f, &wave_a, &[0.1, -0.2, 0.3]).unwrap();
            write_row(&mut f, &wave_b, &[0.4, 0.5, -0.6]).unwrap();
        }

        let (waves, intents, read_cfg) = read_all(path.to_str().unwrap()).unwrap();
        assert_eq!(read_cfg, cfg, "config header");
        assert_eq!(intents.len(), 2, "row count");
        assert_eq!(waves.len(), 2 * WAVE_CHANNELS, "wave payload size");
        assert_eq!(&waves[..WAVE_CHANNELS], &wave_a[..], "row 0 wave");
        assert_eq!(&waves[WAVE_CHANNELS..], &wave_b[..], "row 1 wave");
        assert_eq!(intents[1], [0.4, 0.5, -0.6], "row 1 command");
    }

    #[test]
    fn a_legacy_headerless_file_is_rejected_not_defaulted() {
        let dir = TempDir::new("legacy");
        let path = dir.file("w.f32");
        // Pre-header format: rows straight from byte 0, no magic.
        let wave = vec![0.0f32; WAVE_CHANNELS];
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            write_row(&mut f, &wave, &[0.0, 0.0, 0.0]).unwrap();
        }
        let err = read_all(path.to_str().unwrap()).unwrap_err();
        assert!(
            err.to_string().contains("UWAV"),
            "must name the missing header, got: {err}"
        );
    }

    /// Regression: `if version > FORMAT_VERSION` let version 0 through as if it were v1,
    /// silently accepting a corrupt or hand-edited header instead of naming the problem.
    #[test]
    fn a_zero_version_header_is_rejected_not_read_as_v1() {
        let dir = TempDir::new("v0");
        let path = dir.file("w.f32");
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            f.write_all(MAGIC).unwrap();
            f.write_all(&0u32.to_le_bytes()).unwrap();
            let cfg = sample_config();
            f.write_all(bytemuck::bytes_of(&cfg)).unwrap();
        }
        let err = read_all(path.to_str().unwrap()).unwrap_err();
        assert!(
            err.to_string().contains("version 0"),
            "must name the bad version, got: {err}"
        );
    }

    #[test]
    fn sensor_noise_makes_a_recording_irreproducible_and_it_is_rejected() {
        let dir = TempDir::new("noise");
        let path = dir.file("w.f32");
        let mut cfg = sample_config();
        cfg.noise_sigma = 6.0;
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            write_header(&mut f, &cfg).unwrap();
        }
        let err = read_all(path.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("noise"), "got: {err}");
    }

    #[test]
    fn a_ghost_pilot_recording_is_rejected() {
        let dir = TempDir::new("pilot");
        let path = dir.file("w.f32");
        let mut cfg = sample_config();
        cfg.pilot_active = 1;
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            write_header(&mut f, &cfg).unwrap();
        }
        let err = read_all(path.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("pilot"), "got: {err}");
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_short_read() {
        let dir = TempDir::new("trunc");
        let path = dir.file("w.f32");
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            write_header(&mut f, &sample_config()).unwrap();
            // 195 f32 per row; 100 f32 is half a row.
            f.write_all(&vec![0u8; 100 * 4]).unwrap();
        }
        assert!(
            read_all(path.to_str().unwrap()).is_err(),
            "must reject a partial row"
        );
    }

    /// Regression for the exact class of bug `slow_gather.wgsl`'s own comment records
    /// ("was 56u (223-channel wave) — the slow brain read perception with the wrong
    /// stride, scrambled input"): pack, then unpack with the SAME byte/word math
    /// `PerceptionGpu::read_after_submit` uses, and every channel must survive,
    /// including negative ones (the sign bit is exactly what a stride bug loses first).
    #[test]
    fn packed_wave_survives_the_same_unpack_math_as_read_after_submit() {
        let wave: Vec<f32> = (0..WAVE_CHANNELS)
            .map(|ch| (((ch as i32 * 37 + 11) % 256) - 128) as f32)
            .collect();
        let packed = pack_wave_u32(&wave, 3);
        assert_eq!(packed.len(), 3 * WAVE_U32S);

        for being in 0..3 {
            let base = being * WAVE_U32S;
            for ch in 0..WAVE_CHANNELS {
                let word = packed[base + ch / 4];
                let byte = ((word >> ((ch % 4) * 8)) & 0xFF) as u8;
                let decoded = byte as i8 as f32;
                assert_eq!(decoded, wave[ch], "being {being} channel {ch}");
            }
        }
    }

    #[test]
    fn magic_is_four_bytes() {
        assert_eq!(MAGIC.len(), 4);
    }
}
