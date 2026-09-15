//! On-disk genome format for the taxi world — `TAXG` (SPEC §3.3, plan §3.3).
//!
//! A genome is `NP = NF + S + 1 = 40` `f32` weights (SPEC §0, §4.3). The format
//! carries `nf`/`s` in its header and REFUSES a mismatch rather than aligning by
//! position: `NM_RISCO`/`NM_PONTO` change `NF` on the Python side (SPEC §4.2), so a
//! 27-weight genome silently loaded against a 29-weight build would run — just with
//! every weight shifted onto the wrong sense. Aligning by NAME instead is a
//! separate, Python-side tool (`e43_campo.py:alinha_semente`, plan §3.3); this
//! module only ever accepts an exact match or refuses the file.

use std::io::{self, Read, Write};

use crate::constants::{NF, NP, S};

/// File magic, 4 bytes, ASCII `"TAXG"`.
pub const MAGIC: [u8; 4] = *b"TAXG";
/// Format version this build writes and reads. Bump on any layout change.
pub const FORMAT_VERSION: u32 = 1;

/// Everything that can go wrong loading a `.taxg` file.
#[derive(Debug, thiserror::Error)]
pub enum GenomeError {
    #[error("I/O error reading a genome file: {0}")]
    Io(#[from] io::Error),
    #[error("bad magic: expected {expected:?}, found {found:?} (not a TAXG file)")]
    BadMagic { expected: [u8; 4], found: [u8; 4] },
    #[error("unsupported TAXG format version {found} (this build writes/reads {expected})")]
    UnsupportedVersion { expected: u32, found: u32 },
    #[error(
        "genome file has {found} movement/body senses (`nf`), this build expects {expected} \
         (SPEC §4.2: NM_RISCO/NM_PONTO change NF) — refusing to align by position"
    )]
    SenseCountMismatch { expected: u32, found: u32 },
    #[error("genome file has {found} service weights (`s`), this build expects {expected}")]
    ServiceCountMismatch { expected: u32, found: u32 },
    #[error("genome file declares {declared} genome(s) but only had enough bytes for {available}")]
    Truncated { declared: u32, available: usize },
}

/// One genome: `NP = 40` weights, in `NOMES_F` order (SPEC §4.3) — `NOMES_OP (10)`,
/// `NOMES_CORPO (19)`, 10 service weights, 1 service bias.
#[derive(Debug, Clone, PartialEq)]
pub struct Genome(pub Vec<f32>);

impl Genome {
    /// Writes this genome alone (`n = 1` in the header). Panics if `self.0.len() !=
    /// NP` — a malformed in-memory genome is a programmer error, not a file-format
    /// error, so it is not folded into `GenomeError`.
    pub fn save(&self, mut w: impl Write) -> io::Result<()> {
        assert_eq!(
            self.0.len(),
            NP,
            "genome must have exactly NP={NP} weights, got {}",
            self.0.len()
        );
        w.write_all(&MAGIC)?;
        w.write_all(&FORMAT_VERSION.to_le_bytes())?;
        w.write_all(&(NF as u32).to_le_bytes())?;
        w.write_all(&(S as u32).to_le_bytes())?;
        w.write_all(&1u32.to_le_bytes())?;
        for &weight in &self.0 {
            w.write_all(&weight.to_le_bytes())?;
        }
        Ok(())
    }

    /// Reads every genome from a `.taxg` file. Returns `Err` (not a truncated or
    /// misaligned `Vec`) on a magic/version/`nf`/`s` mismatch or a short file.
    pub fn load(mut r: impl Read) -> Result<Vec<Genome>, GenomeError> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(GenomeError::BadMagic {
                expected: MAGIC,
                found: magic,
            });
        }
        let version = read_u32(&mut r)?;
        if version != FORMAT_VERSION {
            return Err(GenomeError::UnsupportedVersion {
                expected: FORMAT_VERSION,
                found: version,
            });
        }
        let nf = read_u32(&mut r)?;
        if nf != NF as u32 {
            return Err(GenomeError::SenseCountMismatch {
                expected: NF as u32,
                found: nf,
            });
        }
        let s = read_u32(&mut r)?;
        if s != S as u32 {
            return Err(GenomeError::ServiceCountMismatch {
                expected: S as u32,
                found: s,
            });
        }
        let n = read_u32(&mut r)?;

        let mut rest = Vec::new();
        r.read_to_end(&mut rest)?;
        let available = rest.len() / 4;
        let needed = n as usize * NP;
        if available < needed {
            return Err(GenomeError::Truncated {
                declared: n,
                available,
            });
        }

        let mut genomes = Vec::with_capacity(n as usize);
        for g in 0..n as usize {
            let mut weights = Vec::with_capacity(NP);
            for i in 0..NP {
                let off = (g * NP + i) * 4;
                let bytes: [u8; 4] = rest[off..off + 4].try_into().unwrap();
                weights.push(f32::from_le_bytes(bytes));
            }
            genomes.push(Genome(weights));
        }
        Ok(genomes)
    }
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_genome() -> Genome {
        Genome((0..NP).map(|i| i as f32 * 0.1 - 1.0).collect())
    }

    #[test]
    fn genome_survives_disk_round_trip() {
        let original = sample_genome();
        let mut buf = Vec::new();
        original
            .save(&mut buf)
            .expect("save must not fail on a well-formed genome");

        let loaded = Genome::load(Cursor::new(&buf)).expect("load must succeed on what save wrote");
        assert_eq!(loaded.len(), 1, "save always writes exactly one genome");
        assert_eq!(
            loaded[0], original,
            "every weight must round-trip bit for bit, not just approximately"
        );
        // Bit-exactness, not just PartialEq on f32 (which is fine for these finite
        // values, but let's be explicit that this is a byte round-trip, not luck).
        for (a, b) in loaded[0].0.iter().zip(original.0.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn genome_with_the_wrong_sense_count_is_refused() {
        // Hand-crafted header: nf=27 (SPEC §4.2: what you get WITHOUT
        // NM_RISCO/NM_PONTO) against a build compiled for NF=29.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&27u32.to_le_bytes()); // wrong nf
        buf.extend_from_slice(&(S as u32).to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Body length is irrelevant: the header must be rejected before it is read.
        buf.extend_from_slice(&vec![0u8; (27 + S + 1) * 4]);

        let err = Genome::load(Cursor::new(&buf)).expect_err("nf mismatch must be refused");
        match err {
            GenomeError::SenseCountMismatch { expected, found } => {
                assert_eq!(expected, NF as u32);
                assert_eq!(found, 27);
            }
            other => panic!("expected SenseCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_bad_magic_is_refused_before_anything_else_is_read() {
        let buf = b"NOPE0000000000000000".to_vec();
        let err = Genome::load(Cursor::new(&buf)).expect_err("bad magic must be refused");
        assert!(matches!(err, GenomeError::BadMagic { .. }), "got {err:?}");
    }

    #[test]
    fn a_truncated_body_is_refused_not_silently_shortened() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(NF as u32).to_le_bytes());
        buf.extend_from_slice(&(S as u32).to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // claims 2 genomes
        buf.extend_from_slice(&vec![0u8; NP * 4]); // only bytes for 1

        let err = Genome::load(Cursor::new(&buf)).expect_err("declared count must be checked");
        assert!(
            matches!(err, GenomeError::Truncated { declared: 2, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn load_reads_multiple_genomes_back_in_order() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(NF as u32).to_le_bytes());
        buf.extend_from_slice(&(S as u32).to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        let g0: Vec<f32> = (0..NP).map(|i| i as f32).collect();
        let g1: Vec<f32> = (0..NP).map(|i| -(i as f32)).collect();
        for w in g0.iter().chain(g1.iter()) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        let loaded = Genome::load(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, g0);
        assert_eq!(loaded[1].0, g1);
    }
}
