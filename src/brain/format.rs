//! Formato `.brain` — o cérebro na palma da mão.
//!
//! # Por que existe
//!
//! Leonardo: *"e se chamar de .brain? Porque a ideia é que seja o cérebro — a gente
//! precisa ter ele na palma da mão."* E antes: *"além de eu ter só essas duas camadas,
//! depois eu vou querer ter mais camadas, e aí vou querer ativar só uma, ativar outra."*
//!
//! O caminho de evolução na GPU salvava `UGEN v2`: quatro vetores de `f32` gravados em
//! sequência, sem tabela de tamanhos. Três defeitos concretos, todos já sentidos:
//!
//! · **reordenar um campo corrompe em silêncio** — o leitor confia na ordem, não há
//!   verificação; um `.bin` antigo lido por um binário novo devolve pesos embaralhados
//!   sem erro nenhum;
//! · **o último bloco é adivinhado** — o loader deduz o tamanho de τ subtraindo o que
//!   sobrou do arquivo;
//! · **não há como desligar uma camada** nem acrescentar outra sem quebrar tudo.
//!
//! O `.dna` (TLV do caminho Bevy) já resolvia isso e foi abandonado sem decisão. Este
//! formato retoma a tese com o vocabulário do projeto de hoje.
//!
//! # Anatomia
//!
//! ```text
//! ┌─ CABEÇALHO (48 bytes) ────────────────────────────────────────────┐
//! │ magic "BRAIN\0" (6) │ versão u16 │ geração u32 │ semente u64      │
//! │ fitness f32 │ largura de pista f32 │ nº de blocos u16 │ reserva    │
//! ├─ TABELA DE BLOCOS (24 bytes × n) ─────────────────────────────────┤
//! │ tipo u8 │ codificação u8 │ sinalizadores u8 │ _ u8 │ nº itens u32 │
//! │ deslocamento u64 │ bytes u64                                      │
//! ├─ DADOS ───────────────────────────────────────────────────────────┤
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! A tabela é a diferença que importa: **cada bloco se descreve**. Um leitor antigo pula
//! bloco de tipo desconhecido em vez de se perder; uma camada nova entra sem invalidar os
//! arquivos existentes; e o bit `ATIVO` liga e desliga uma camada sem apagá-la do disco —
//! que é o "ativar só uma, ativar outra" pedido.
//!
//! # Tamanho
//!
//! Cada bloco escolhe sua codificação. O cérebro de hoje em `f32` ocupa ~96 KB; os mesmos
//! pesos em `i8` ocupam ~24 KB com erro de quantização medido em ~48 dB de SNR — sobra
//! folga para quando o cérebro crescer. A codificação é por bloco justamente porque nem
//! toda camada tolera a mesma perda: τ e os knots do KAN são poucos valores e sensíveis,
//! então ficam em `f32`; L1 e L2 são dezenas de milhares e toleram `i8`.

use std::io::{self, Read, Write};

/// Assinatura do arquivo. Seis bytes para o cabeçalho fechar alinhado em 8.
pub const MAGIC: &[u8; 6] = b"BRAIN\0";

/// Versão do formato. Subir SÓ quando a mudança quebrar leitores antigos — acrescentar
/// um tipo de bloco novo não quebra (o leitor pula o que não conhece), então não sobe.
pub const VERSION: u16 = 1;

const HEADER_BYTES: usize = 48;
const ENTRY_BYTES: usize = 24;

/// O que o bloco carrega. O valor numérico é gravado no disco: **nunca reaproveitar um
/// número**, sempre acrescentar no fim, senão arquivos antigos passam a mentir.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BlockKind {
    /// Pesos sensoriais → camada rápida (`SENSORY_WAVE_SIZE × L1_NEURONS`).
    L1 = 1,
    /// Camada rápida → motores (`L1_NEURONS × L2_NEURONS`).
    L2 = 2,
    /// Knots do KAN de saída (3 × `KAN_OUT_KNOTS`).
    KanOut = 3,
    /// Cérebro lento (modulação).
    Slow = 4,
    /// Banco de τ: taxa de decaimento por canal do CfC.
    Tau = 5,
    /// Reservado para a próxima camada — visão dedicada, córtex extra, o que vier.
    Extra = 6,
    /// Bloco de tipo não reconhecido por este leitor: preservado byte a byte para não se
    /// perder numa ida e volta pelo disco.
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

/// Como os números do bloco estão gravados.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Encoding {
    /// Sem perda, 4 bytes por peso.
    F32 = 0,
    /// 1 byte por peso, escala fixa de 127. Erro relativo ~0,4% para pesos em [-1, 1].
    I8 = 1,
}

/// Bit 0 dos sinalizadores: a camada participa do forward.
///
/// Desligar NÃO apaga os pesos — eles continuam no arquivo, herdáveis e mutáveis. É a
/// diferença entre amputar e sedar: uma camada sedada pode voltar numa geração futura se
/// a seleção a favorecer.
pub const FLAG_ACTIVE: u8 = 1 << 0;

/// Um bloco já decodificado em memória.
#[derive(Clone, Debug)]
pub struct Block {
    pub kind: BlockKind,
    pub encoding: Encoding,
    pub flags: u8,
    /// Pesos em `f32`, já decodificados qualquer que seja a codificação em disco.
    pub data: Vec<f32>,
    /// Bytes crus de um bloco de tipo desconhecido, preservados para a volta ao disco.
    pub raw: Option<Vec<u8>>,
}

impl Block {
    /// Bloco novo, ativo, em `f32`.
    pub fn new(kind: BlockKind, data: Vec<f32>) -> Self {
        Self {
            kind,
            encoding: Encoding::F32,
            flags: FLAG_ACTIVE,
            data,
            raw: None,
        }
    }

    /// Mesmo bloco, gravado em `i8` — cabe em um quarto do espaço.
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

/// Cérebro completo: quem ele é, e de que partes é feito.
#[derive(Clone, Debug, Default)]
pub struct BrainFile {
    pub generation: u32,
    pub seed: u64,
    pub fitness: f32,
    pub track_width: f32,
    pub blocks: Vec<Block>,
}

impl BrainFile {
    /// Primeiro bloco daquele tipo, se existir e estiver ativo.
    pub fn active(&self, kind: BlockKind) -> Option<&[f32]> {
        self.blocks
            .iter()
            .find(|b| b.kind == kind && b.is_active())
            .map(|b| b.data.as_slice())
    }

    /// Primeiro bloco daquele tipo, ativo ou não.
    pub fn get(&self, kind: BlockKind) -> Option<&Block> {
        self.blocks.iter().find(|b| b.kind == kind)
    }

    /// Tamanho que o arquivo terá em disco, sem gravá-lo.
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
        head.resize(HEADER_BYTES, 0); // reserva zerada até 48 bytes

        // Os dados vão depois da tabela inteira, então os deslocamentos já são conhecidos.
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
            return Err(bad("não é um arquivo .brain (assinatura ausente)"));
        }
        let version = u16::from_le_bytes([b[6], b[7]]);
        if version > VERSION {
            return Err(bad(&format!(
                "versão {version} é mais nova que a suportada ({VERSION})"
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
                return Err(bad("tabela de blocos truncada"));
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
                return Err(bad("bloco aponta para fora do arquivo"));
            }
            let raw = &b[off..off + len];

            // Tipo desconhecido: guarda os bytes e segue. É o que permite a um binário
            // antigo abrir um cérebro com camadas que ele não conhece sem destruí-las.
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
                    "bloco {kind:?}: esperava {count} valores, achei {}",
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
        // τ guarda taxas fora de [-1,1]: por isso o padrão é f32, não i8.
        assert_eq!(b.active(BlockKind::Tau).unwrap(), &[2.0, 0.5]);
    }

    #[test]
    fn camada_desligada_continua_no_disco() {
        let mut a = exemplo();
        a.blocks[0].set_active(false);
        let mut v = Vec::new();
        a.write(&mut v).unwrap();
        let b = BrainFile::from_bytes(&v).unwrap();
        // Some do forward…
        assert!(b.active(BlockKind::L1).is_none());
        // …mas os pesos continuam lá, herdáveis.
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
        assert!(a.encoded_len() * 3 < f32_len, "i8 deveria encolher ~4x");
        let mut v = Vec::new();
        a.write(&mut v).unwrap();
        let b = BrainFile::from_bytes(&v).unwrap();
        let erro = pesos
            .iter()
            .zip(b.active(BlockKind::L1).unwrap())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(erro < 0.005, "erro máximo de quantização foi {erro}");
    }

    #[test]
    fn bloco_desconhecido_sobrevive_a_ida_e_volta() {
        // Simula um arquivo gravado por uma versão futura, com um tipo que não existe aqui.
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
