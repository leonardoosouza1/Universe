//! Métricas de ativação e guardas de runtime (E4 — REFACTOR_PLAN.md §E4).
//!
//! Consome só `math::*` (E1) e tipos primitivos — nenhuma dependência de GPU/Bevy
//! aqui dentro. Quem lê os buffers da GPU e monta os `Vec<f32>` de amostra é o
//! caller (`src/bin/gpu_evolve/context.rs`); este módulo só enxerga números.
pub mod activation;
