//! Shim: o conteúdo real mudou para `universe::gpu::genome_io` (lib) em 2026-08-14
//! para o parser canônico poder ser usado por `genoma_view` e futuros leitores —
//! ver o header lá pelo bug que motivou (parser duplicado lendo τ errado).
pub use universe::gpu::genome_io::*;
