//! Raw sensory-wave recording: the input side of the open-loop screen.
//!
//! `obs.f32` is NOT enough for this. Its columns 10..202 are post-KAN and post-CfC —
//! already transformed by the recording genome's own KAN knots and tau bank. Replaying
//! a DIFFERENT genome from them would hold those stages fixed and only vary L1/L2. The
//! raw 192-channel wave is the last point in the pipeline that is genome-independent.

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

/// Ticks recorded so far. Reserved for the writer side (`--gravar-onda`) to report a
/// row count without re-deriving it from bytes written; not yet read by anything.
pub struct WaveRecord {
    pub ticks: usize,
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

#[cfg(test)]
mod tests {
    use super::{WAVE_CHANNELS, read_all, write_row};

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
        assert!(
            read_all(path.to_str().unwrap()).is_err(),
            "must reject a partial row"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
