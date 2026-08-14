//! Camada de segurança estilo Tesla — regras puras, sem rede neural (E6: doc adicionada
//! na revisão de legibilidade final; módulo em si não foi tocado nesta rodada).
//!
//! `Sentinel` fica ENTRE a saída do cérebro (`IntentVec` proposto) e o atuador: Tier 0 é
//! veto DURO (substitui a intenção inteira, ex. colisão iminente), Tier 1/2 são propostas
//! SUAVES que o `gate` pondera. Consumido hoje só pelo caminho Bevy legado
//! (`src/engine.rs`, `src/simulation/motor.rs`) — o caminho GPU (`gpu_evolve`) não passa
//! por aqui ainda.
pub mod sentinel;
pub use sentinel::{Sentinel, SafetyState, IntentVec, SentinelOutput, SafetyFlags};
