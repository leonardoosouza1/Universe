//! DNA byte-format checks only: no forward pass, body dynamics, or device.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct TempFile(std::path::PathBuf);
impl TempFile {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "universe-organism-dna-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn fixture() -> Genome {
    let mut genome = Genome::random(1701);
    genome.fly = vec![0.125, -0.25, 0.5];
    genome
}
fn encode(genome: &Genome) -> Vec<u8> {
    let file = TempFile::new();
    save_genome(genome, file.path()).unwrap();
    std::fs::read(&file.0).unwrap()
}
fn decode(bytes: &[u8]) -> Option<Genome> {
    let file = TempFile::new();
    std::fs::write(&file.0, bytes).unwrap();
    load_genome(file.path())
}

#[test]
fn v5_roundtrip_keeps_exact_v2_payload_and_hereditary_extensions() {
    let mut genome = fixture();
    let legacy_payload = encode(&genome)[8..].to_vec();
    genome.body.0 = [0.25, -0.5];
    genome.organs.0 = [-0.25, 0.5];
    genome.organism = std::array::from_fn(|i| (i as f32 - 8.0) / 8.0);
    let bytes = encode(&genome);
    assert_eq!(&bytes[..8], b"UGEN\x05\x00\x00\x00");
    let header = 12 + (2 + 2 + N_LIFE_GENES + 16) * 4;
    assert_eq!(&bytes[header..], legacy_payload);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
        legacy_payload.len()
    );
    let loaded = decode(&bytes).unwrap();
    assert_eq!(loaded.organism, genome.organism);
    assert_eq!(loaded.body, genome.body);
    assert_eq!(loaded.organs, genome.organs);
    assert_eq!(loaded.lifetime, genome.lifetime);
    assert_eq!(encode(&loaded), bytes);
}

#[test]
fn legacy_v2_v3_v4_roundtrip_stays_byte_exact_and_defaults_organism() {
    for version in 2u32..=4 {
        let mut genome = fixture();
        if version >= 3 {
            genome.body.0[0] = 0.25;
        }
        if version >= 4 {
            genome.organs.0[0] = 0.25;
        }
        let bytes = encode(&genome);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), version);
        let loaded = decode(&bytes).unwrap();
        assert_eq!(loaded.organism, [0.0; 16]);
        assert_eq!(encode(&loaded), bytes);
    }
}

#[test]
fn v5_rejects_invalid_domain_version_lengths_and_nonfinite_payload() {
    let mut genome = fixture();
    genome.organism[0] = 0.5;
    let bytes = encode(&genome);
    for end in [0, 4, 7, 11, 91, 155, bytes.len() - 1, bytes.len() - 4] {
        assert!(
            decode(&bytes[..end]).is_none(),
            "accepted truncated length {end}"
        );
    }
    for version in [0u32, 1, 6, u32::MAX] {
        let mut bad = bytes.clone();
        bad[4..8].copy_from_slice(&version.to_le_bytes());
        assert!(decode(&bad).is_none());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.01, 1.01] {
        let mut bad = bytes.clone();
        let offset = 12 + (2 + 2 + N_LIFE_GENES) * 4;
        bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(decode(&bad).is_none());
        genome.organism[0] = value;
        let target = TempFile::new();
        assert!(save_genome(&genome, target.path()).is_err());
        assert!(
            !target.0.exists(),
            "invalid DNA must not create a checkpoint"
        );
    }
    let mut bad = bytes.clone();
    bad.extend_from_slice(&0.0f32.to_le_bytes());
    assert!(decode(&bad).is_none());
    let mut bad = bytes;
    let payload = 12 + (2 + 2 + N_LIFE_GENES + 16) * 4;
    bad[payload..payload + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(decode(&bad).is_none());
}

#[test]
fn v5_writer_rejects_malformed_neural_payload_before_touching_file() {
    let mut genome = fixture();
    genome.organism[0] = 0.25;
    for corrupt_shape in [false, true] {
        let mut bad = genome.clone();
        if corrupt_shape {
            bad.l1.pop();
        } else {
            bad.l1[0] = f32::INFINITY;
        }
        let file = TempFile::new();
        assert!(save_genome(&bad, file.path()).is_err());
        assert!(!file.0.exists());
    }
}
