//! `.brain` format — the brain in the palm of your hand.
//!
//! # Why it exists
//!
//! Leonardo: *"what if we call it .brain? Because the idea is that it IS the brain — we
//! need to have it in the palm of our hand."* And earlier: *"besides having just these
//! two layers, later I'll want more layers, and then I'll want to activate just one,
//! activate another."*
//!
//! The GPU evolution path saved `UGEN v2`: four `f32` vectors written in sequence,
//! with no size table. Three concrete defects, all already felt:
//!
//! · **reordering a field corrupts silently** — the reader trusts the order, there is no
//!   verification; an old `.bin` read by a new binary returns shuffled weights
//!   with no error at all;
//! · **the last block is guessed** — the loader deduces τ's size by subtracting what
//!   is left of the file;
//! · **there is no way to turn a layer off** nor add another without breaking everything.
//!
//! The `.dna` (TLV of the Bevy path) already solved this and was abandoned without a
//! decision. This format resumes the thesis with today's project vocabulary.
//!
//! # Anatomy
//!
//! ```text
//! ┌─ HEADER (48 bytes) ───────────────────────────────────────────────┐
//! │ magic "BRAIN\0" (6) │ version u16 │ generation u32 │ seed u64     │
//! │ fitness f32 │ track width f32 │ block count u16 │ reserved        │
//! ├─ BLOCK TABLE (24 bytes × n) ──────────────────────────────────────┤
//! │ kind u8 │ encoding u8 │ flags u8 │ _ u8 │ item count u32          │
//! │ offset u64 │ bytes u64                                            │
//! ├─ DATA ────────────────────────────────────────────────────────────┤
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The table is the difference that matters: **each block describes itself**. An old
//! reader skips a block of unknown kind instead of getting lost; a new layer comes in
//! without invalidating existing files; and the `ACTIVE` bit turns a layer on and off
//! without erasing it from disk — which is the requested "activate just one, activate
//! another".
//!
//! # Size
//!
//! Each block picks its encoding. Today's brain in `f32` takes ~96 KB; the same
//! weights in `i8` take ~24 KB with quantization error measured at ~48 dB of SNR —
//! plenty of headroom for when the brain grows. Encoding is per block precisely because
//! not every layer tolerates the same loss: τ and the KAN knots are few, sensitive
//! values, so they stay in `f32`; L1 and L2 are tens of thousands and tolerate `i8`.

use std::io::{self, Read, Write};

/// File signature. Six bytes so the header closes aligned to 8.
pub const MAGIC: &[u8; 6] = b"BRAIN\0";

/// Format version. Bump ONLY when the change breaks old readers — adding a new
/// block kind does not break (the reader skips what it does not know), so no bump.
pub const VERSION: u16 = 1;

const HEADER_BYTES: usize = 48;
const ENTRY_BYTES: usize = 24;

/// What the block carries. The numeric value is written to disk: **never reuse a
/// number**, always append at the end, or old files start to lie.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BlockKind {
    /// Sensory weights → fast layer (`SENSORY_WAVE_SIZE × L1_NEURONS`).
    L1 = 1,
    /// Fast layer → motors (`L1_NEURONS × L2_NEURONS`).
    L2 = 2,
    /// Output KAN knots (3 × `KAN_OUT_KNOTS`).
    KanOut = 3,
    /// Slow brain (modulation).
    Slow = 4,
    /// τ bank: decay rate per CfC channel.
    Tau = 5,
    /// Reserved for the next layer — dedicated vision, extra cortex, whatever comes.
    Extra = 6,
    /// Block of a kind this reader does not recognize: preserved byte for byte so it is
    /// not lost on a round trip through disk.
    Unknown = 255,
}

impl BlockKind {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::L1,
            2 => Self::L2,
            3 => Self::KanOut,
            4 => Self::Slow,
            5 => Self::Tau,
            6 => Self::Extra,
            _ => Self::Unknown,
        }
    }
}

/// How the block's numbers are written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Encoding {
    /// Lossless, 4 bytes per weight.
    F32 = 0,
    /// 1 byte per weight, fixed scale of 127. Relative error ~0.4% for weights in [-1, 1].
    I8 = 1,
}

/// Bit 0 of the flags: the layer takes part in the forward.
///
/// Turning it off does NOT erase the weights — they stay in the file, heritable and
/// mutable. It is the difference between amputating and sedating: a sedated layer can
/// come back in a future generation if selection favors it.
pub const FLAG_ACTIVE: u8 = 1 << 0;

/// A block already decoded in memory.
#[derive(Clone, Debug)]
pub struct Block {
    pub kind: BlockKind,
    pub encoding: Encoding,
    pub flags: u8,
    /// Weights in `f32`, already decoded whatever the on-disk encoding.
    pub data: Vec<f32>,
    /// Raw bytes of an unknown-kind block, preserved for the trip back to disk.
    pub raw: Option<Vec<u8>>,
}

impl Block {
    /// New block, active, in `f32`.
    pub fn new(kind: BlockKind, data: Vec<f32>) -> Self {
        Self {
            kind,
            encoding: Encoding::F32,
            flags: FLAG_ACTIVE,
            data,
            raw: None,
        }
    }

    /// Same block, written in `i8` — fits in a quarter of the space.
    pub fn compact(mut self) -> Self {
        self.encoding = Encoding::I8;
        self
    }

    pub fn is_active(&self) -> bool {
        self.flags & FLAG_ACTIVE != 0
    }

    pub fn set_active(&mut self, on: bool) {
        if on {
            self.flags |= FLAG_ACTIVE
        } else {
            self.flags &= !FLAG_ACTIVE
        }
    }
}

/// Complete brain: who it is, and what parts it is made of.
#[derive(Clone, Debug, Default)]
pub struct BrainFile {
    pub generation: u32,
    pub seed: u64,
    pub fitness: f32,
    pub track_width: f32,
    pub blocks: Vec<Block>,
}

impl BrainFile {
    /// First block of that kind, if it exists and is active.
    pub fn active(&self, kind: BlockKind) -> Option<&[f32]> {
        self.blocks
            .iter()
            .find(|b| b.kind == kind && b.is_active())
            .map(|b| b.data.as_slice())
    }

    /// First block of that kind, active or not.
    pub fn get(&self, kind: BlockKind) -> Option<&Block> {
        self.blocks.iter().find(|b| b.kind == kind)
    }

    /// Size the file will have on disk, without writing it.
    pub fn encoded_len(&self) -> usize {
        HEADER_BYTES
            + ENTRY_BYTES * self.blocks.len()
            + self
                .blocks
                .iter()
                .map(|b| match (b.encoding, &b.raw) {
                    (_, Some(r)) => r.len(),
                    (Encoding::F32, _) => b.data.len() * 4,
                    (Encoding::I8, _) => b.data.len(),
                })
                .sum::<usize>()
    }

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut head = Vec::with_capacity(HEADER_BYTES + ENTRY_BYTES * self.blocks.len());
        head.extend_from_slice(MAGIC);
        head.extend_from_slice(&VERSION.to_le_bytes());
        head.extend_from_slice(&self.generation.to_le_bytes());
        head.extend_from_slice(&self.seed.to_le_bytes());
        head.extend_from_slice(&self.fitness.to_le_bytes());
        head.extend_from_slice(&self.track_width.to_le_bytes());
        head.extend_from_slice(&(self.blocks.len() as u16).to_le_bytes());
        head.resize(HEADER_BYTES, 0); // zeroed reserve up to 48 bytes

        // The data goes after the whole table, so the offsets are already known.
        let mut off = (HEADER_BYTES + ENTRY_BYTES * self.blocks.len()) as u64;
        let mut body = Vec::new();
        for b in &self.blocks {
            let bytes: Vec<u8> = match (&b.raw, b.encoding) {
                (Some(r), _) => r.clone(),
                (None, Encoding::F32) => b.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                (None, Encoding::I8) => b
                    .data
                    .iter()
                    .map(|v| (v.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8)
                    .collect(),
            };
            head.push(b.kind as u8);
            head.push(b.encoding as u8);
            head.push(b.flags);
            head.push(0);
            head.extend_from_slice(&(b.data.len() as u32).to_le_bytes());
            head.extend_from_slice(&off.to_le_bytes());
            head.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            off += bytes.len() as u64;
            body.extend_from_slice(&bytes);
        }
        w.write_all(&head)?;
        w.write_all(&body)
    }

    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        Self::from_bytes(&buf)
    }

    pub fn from_bytes(b: &[u8]) -> io::Result<Self> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if b.len() < HEADER_BYTES || &b[0..6] != MAGIC {
            return Err(bad("not a .brain file (signature missing)"));
        }
        let version = u16::from_le_bytes([b[6], b[7]]);
        if version > VERSION {
            return Err(bad(&format!(
                "version {version} is newer than the supported one ({VERSION})"
            )));
        }
        let generation = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let seed = u64::from_le_bytes(b[12..20].try_into().unwrap());
        let fitness = f32::from_le_bytes(b[20..24].try_into().unwrap());
        let track_width = f32::from_le_bytes(b[24..28].try_into().unwrap());
        let n = u16::from_le_bytes([b[28], b[29]]) as usize;

        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let e = HEADER_BYTES + i * ENTRY_BYTES;
            if e + ENTRY_BYTES > b.len() {
                return Err(bad("block table truncated"));
            }
            let kind = BlockKind::from_u8(b[e]);
            let encoding = if b[e + 1] == 1 {
                Encoding::I8
            } else {
                Encoding::F32
            };
            let flags = b[e + 2];
            let count = u32::from_le_bytes(b[e + 4..e + 8].try_into().unwrap()) as usize;
            let off = u64::from_le_bytes(b[e + 8..e + 16].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(b[e + 16..e + 24].try_into().unwrap()) as usize;
            if off + len > b.len() {
                return Err(bad("block points outside the file"));
            }
            let raw = &b[off..off + len];

            // Unknown kind: keep the bytes and move on. This is what lets an old binary
            // open a brain with layers it does not know without destroying them.
            if kind == BlockKind::Unknown {
                blocks.push(Block {
                    kind,
                    encoding,
                    flags,
                    data: Vec::new(),
                    raw: Some(raw.to_vec()),
                });
                continue;
            }
            let data: Vec<f32> = match encoding {
                Encoding::F32 => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                Encoding::I8 => raw.iter().map(|&v| v as i8 as f32 / 127.0).collect(),
            };
            if data.len() != count {
                return Err(bad(&format!(
                    "block {kind:?}: expected {count} values, found {}",
                    data.len()
                )));
            }
            blocks.push(Block {
                kind,
                encoding,
                flags,
                data,
                raw: None,
            });
        }
        Ok(Self {
            generation,
            seed,
            fitness,
            track_width,
            blocks,
        })
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.write(&mut f)
    }

    pub fn load(path: &str) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exemplo() -> BrainFile {
        BrainFile {
            generation: 29,
            seed: 77,
            fitness: 4223.5,
            track_width: 28.0,
            blocks: vec![
                Block::new(BlockKind::L1, vec![0.5, -0.25, 0.125, 1.0]),
                Block::new(BlockKind::Tau, vec![2.0, 0.5]),
            ],
        }
    }

    #[test]
    fn ida_e_volta_preserva_tudo() {
        let a = exemplo();
        let mut v = Vec::new();
        a.write(&mut v).unwrap();
        let b = BrainFile::from_bytes(&v).unwrap();
        assert_eq!(b.generation, 29);
        assert_eq!(b.seed, 77);
        assert_eq!(b.track_width, 28.0);
        assert_eq!(b.blocks.len(), 2);
        assert_eq!(b.active(BlockKind::L1).unwrap(), &[0.5, -0.25, 0.125, 1.0]);
        // τ holds rates outside [-1,1]: that is why the default is f32, not i8.
        assert_eq!(b.active(BlockKind::Tau).unwrap(), &[2.0, 0.5]);
    }

    #[test]
    fn camada_desligada_continua_no_disco() {
        let mut a = exemplo();
        a.blocks[0].set_active(false);
        let mut v = Vec::new();
        a.write(&mut v).unwrap();
        let b = BrainFile::from_bytes(&v).unwrap();
        // Disappears from the forward…
        assert!(b.active(BlockKind::L1).is_none());
        // …but the weights are still there, heritable.
        assert_eq!(b.get(BlockKind::L1).unwrap().data.len(), 4);
    }

    #[test]
    fn i8_encolhe_quatro_vezes_com_erro_pequeno() {
        let pesos: Vec<f32> = (0..1000).map(|i| (i as f32 / 500.0 - 1.0) * 0.9).collect();
        let f32_len = BrainFile {
            blocks: vec![Block::new(BlockKind::L1, pesos.clone())],
            ..Default::default()
        }
        .encoded_len();
        let a = BrainFile {
            blocks: vec![Block::new(BlockKind::L1, pesos.clone()).compact()],
            ..Default::default()
        };
        assert!(a.encoded_len() * 3 < f32_len, "i8 should shrink ~4x");
        let mut v = Vec::new();
        a.write(&mut v).unwrap();
        let b = BrainFile::from_bytes(&v).unwrap();
        let erro = pesos
            .iter()
            .zip(b.active(BlockKind::L1).unwrap())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(erro < 0.005, "maximum quantization error was {erro}");
    }

    #[test]
    fn bloco_desconhecido_sobrevive_a_ida_e_volta() {
        // Simulates a file written by a future version, with a kind that does not exist here.
        let mut a = exemplo();
        a.blocks.push(Block {
            kind: BlockKind::Unknown,
            encoding: Encoding::F32,
            flags: FLAG_ACTIVE,
            data: vec![],
            raw: Some(vec![1, 2, 3, 4]),
        });
        let mut v = Vec::new();
        a.write(&mut v).unwrap();
        let b = BrainFile::from_bytes(&v).unwrap();
        let desconhecido = b
            .blocks
            .iter()
            .find(|x| x.kind == BlockKind::Unknown)
            .unwrap();
        assert_eq!(desconhecido.raw.as_deref(), Some(&[1u8, 2, 3, 4][..]));
    }

    #[test]
    fn recusa_arquivo_que_nao_e_brain() {
        assert!(BrainFile::from_bytes(&[0u8; 64]).is_err());
    }
}
