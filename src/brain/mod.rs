//! Cérebro — formato de arquivo e (futuramente) a topologia em si.
//!
//! Hoje abriga só o formato `.brain`. O módulo existe separado porque a intenção é que
//! ele receba também a definição das camadas: quando o cérebro passar a ter regiões
//! (visão, decisão, motor), a descrição de cada região mora aqui, ao lado do formato que
//! a grava.
pub mod format;

pub use format::{BrainFile, Block, BlockKind, Encoding, FLAG_ACTIVE};
