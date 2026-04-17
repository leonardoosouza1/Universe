use std::fs::File;
use std::io::{Read, Write};

#[derive(Clone)]
pub struct EliteDna {
    pub fast_weights: Vec<i8>,
    pub fast_scale: f32,
    pub slow_weights: Vec<i8>,
    pub slow_scale: f32,
    pub fitness: f32,
}

// ============================================================================
// DNA FILE FORMAT v3
// ----------------------------------------------------------------------------
// Layout (little-endian, 8478 bytes total):
//   magic[4]     = b"IARA"
//   version:u16  = 3
//   generation:u32
//   seed:u32
//   track_width:f32
//   fitness:f32
//   fast_scale:f32
//   fast_weights[8192] (i8 as u8)
//   slow_scale:f32
//   slow_weights[256]  (i8 as u8)
//
// Backward-compat: legacy v1 (8196B) and v2 (8456B) files are detected by
// absence of magic bytes and loaded with zeroed metadata.
// ============================================================================
pub const DNA_MAGIC: &[u8; 4] = b"IARA";
pub const DNA_VERSION: u16 = 3;
pub const DNA_FAST_SIZE: usize = 128 * 64;
pub const DNA_SLOW_SIZE: usize = 32 * 8;
pub const DNA_HEADER_SIZE: usize = 4 + 2 + 4 + 4 + 4 + 4;  // 22
pub const DNA_V3_TOTAL: usize = DNA_HEADER_SIZE + 4 + DNA_FAST_SIZE + 4 + DNA_SLOW_SIZE;  // 8478

pub struct DnaPayload {
    pub fast_weights: Vec<i8>,
    pub fast_scale: f32,
    pub slow_weights: Option<Vec<i8>>,
    pub slow_scale: f32,
    pub generation: u32,
    pub seed: u32,
    pub track_width: f32,
    pub fitness: f32,
    pub version: u16,
}

pub fn save_champion_dna(
    path: &str,
    fast_weights: &[i8],
    fast_scale: f32,
    slow_weights: &[i8],
    slow_scale: f32,
    generation: u32,
    seed: u32,
    track_width: f32,
    fitness: f32,
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(DNA_MAGIC)?;
    file.write_all(&DNA_VERSION.to_le_bytes())?;
    file.write_all(&generation.to_le_bytes())?;
    file.write_all(&seed.to_le_bytes())?;
    file.write_all(&track_width.to_le_bytes())?;
    file.write_all(&fitness.to_le_bytes())?;
    file.write_all(&fast_scale.to_le_bytes())?;
    file.write_all(&fast_weights.iter().map(|&w| w as u8).collect::<Vec<u8>>())?;
    file.write_all(&slow_scale.to_le_bytes())?;
    file.write_all(&slow_weights.iter().map(|&w| w as u8).collect::<Vec<u8>>())?;
    Ok(())
}

pub fn load_champion_dna(path: &str) -> Option<DnaPayload> {
    let mut file = File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;

    // --- v3 with magic ---
    if buf.len() >= DNA_V3_TOTAL && &buf[0..4] == DNA_MAGIC {
        let version = u16::from_le_bytes(buf[4..6].try_into().ok()?);
        let generation = u32::from_le_bytes(buf[6..10].try_into().ok()?);
        let seed = u32::from_le_bytes(buf[10..14].try_into().ok()?);
        let track_width = f32::from_le_bytes(buf[14..18].try_into().ok()?);
        let fitness = f32::from_le_bytes(buf[18..22].try_into().ok()?);
        let fast_scale = f32::from_le_bytes(buf[22..26].try_into().ok()?);
        let fast_end = 26 + DNA_FAST_SIZE;
        let fast_weights: Vec<i8> = buf[26..fast_end].iter().map(|&b| b as i8).collect();
        let slow_scale = f32::from_le_bytes(buf[fast_end..fast_end + 4].try_into().ok()?);
        let slow_weights: Vec<i8> = buf[fast_end + 4..].iter().map(|&b| b as i8).collect();
        return Some(DnaPayload {
            fast_weights, fast_scale,
            slow_weights: Some(slow_weights), slow_scale,
            generation, seed, track_width, fitness, version,
        });
    }

    // --- Legacy v1/v2 (no magic) ---
    if buf.len() >= 4 + DNA_FAST_SIZE {
        let fast_scale = f32::from_le_bytes(buf[0..4].try_into().ok()?);
        let fast_end = 4 + DNA_FAST_SIZE;
        let fast_weights: Vec<i8> = buf[4..fast_end].iter().map(|&b| b as i8).collect();
        let (slow_weights, slow_scale, version) = if buf.len() >= fast_end + 4 + DNA_SLOW_SIZE {
            let ss = f32::from_le_bytes(buf[fast_end..fast_end + 4].try_into().ok()?);
            let sw: Vec<i8> = buf[fast_end + 4..].iter().map(|&b| b as i8).collect();
            (Some(sw), ss, 2u16)
        } else {
            (None, 1.0, 1u16)
        };
        return Some(DnaPayload {
            fast_weights, fast_scale,
            slow_weights, slow_scale,
            generation: 0, seed: 0, track_width: 0.0, fitness: 0.0, version,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn sample_weights(offset: i8) -> (Vec<i8>, Vec<i8>) {
        let fast: Vec<i8> = (0..DNA_FAST_SIZE).map(|i| (((i as i32 + offset as i32) % 3) - 1) as i8).collect();
        let slow: Vec<i8> = (0..DNA_SLOW_SIZE).map(|i| (((i as i32 + offset as i32) % 3) - 1) as i8).collect();
        (fast, slow)
    }

    #[test]
    fn v3_round_trip_preserves_everything() {
        let (fast, slow) = sample_weights(0);
        let path = std::env::temp_dir().join("iara_test_v3.dna");
        let path_s = path.to_str().unwrap();
        save_champion_dna(path_s, &fast, 0.1234, &slow, 0.5678, 42, 1337, 14.5, 999.9).unwrap();
        let loaded = load_champion_dna(path_s).expect("round trip must succeed");
        assert_eq!(loaded.version, DNA_VERSION);
        assert_eq!(loaded.generation, 42);
        assert_eq!(loaded.seed, 1337);
        assert!((loaded.track_width - 14.5).abs() < 1e-5);
        assert!((loaded.fitness - 999.9).abs() < 1e-3);
        assert!((loaded.fast_scale - 0.1234).abs() < 1e-5);
        assert!((loaded.slow_scale - 0.5678).abs() < 1e-5);
        assert_eq!(loaded.fast_weights, fast);
        assert_eq!(loaded.slow_weights.unwrap(), slow);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn legacy_v2_loads_with_zero_metadata() {
        let (fast, slow) = sample_weights(1);
        let path = std::env::temp_dir().join("iara_test_v2.dna");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0.5_f32.to_le_bytes()).unwrap();
        f.write_all(&fast.iter().map(|&w| w as u8).collect::<Vec<u8>>()).unwrap();
        f.write_all(&0.25_f32.to_le_bytes()).unwrap();
        f.write_all(&slow.iter().map(|&w| w as u8).collect::<Vec<u8>>()).unwrap();
        drop(f);
        let loaded = load_champion_dna(path.to_str().unwrap()).expect("v2 must load");
        assert_eq!(loaded.version, 2);
        assert_eq!(loaded.generation, 0);
        assert_eq!(loaded.seed, 0);
        assert_eq!(loaded.fast_weights, fast);
        assert_eq!(loaded.slow_weights.unwrap(), slow);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn missing_file_returns_none() {
        let path = std::env::temp_dir().join("iara_does_not_exist.dna");
        let _ = std::fs::remove_file(&path);
        assert!(load_champion_dna(path.to_str().unwrap()).is_none());
    }

    #[test]
    fn v3_total_size_matches_layout() {
        assert_eq!(DNA_V3_TOTAL, 4 + 2 + 4 + 4 + 4 + 4 + 4 + DNA_FAST_SIZE + 4 + DNA_SLOW_SIZE);
    }
}
