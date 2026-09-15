//! D6 of the difficulty ladder — CHAMPION'S GHOST (see `docs/ESCADA_DIFICULDADE.md`,
//! line `| D6 | fantasma do campeão anterior na pista | --fantasma | spec |`).
//!
//! Live per-sector time pressure, **without physical contact**: the previous
//! generation's champion is not drawn on the track and collides with no one — it becomes
//! a trail of "how long it took to reach each sector", and each being of the current
//! generation is compared against that trail at the SAME sector. There is no x/y
//! position, only sector × tick.
//!
//! ## The pressure's cross currency (why `bonus_vida`, not `penalidade_vida`)
//!
//! Law of the Cross Currency (validated in the ecosystem, see
//! `types.rs::BeingState::life_left` and `CLAUDE.md`): every protective action must
//! cost/pay in a currency DIFFERENT from the one being measured, otherwise the
//! degenerate policy dominates. Here the ghost's pressure is measured in TIME
//! (sector × tick) but the effect is paid in LIFE — not in direct fitness and not in
//! distance — exactly the same choice that already worked for the wall
//! (cost in time, not in meters).
//!
//! Within that choice, the vector is ASYMMETRIC on purpose: being ahead of the
//! ghost GAINS extra life; being behind loses NOTHING beyond what normal earned-life
//! (`--life-earn`, gains life per meter) already charges. Stacking a second
//! punishment on top of the same correlation (slow = behind the ghost = earns little
//! life-per-meter) is a double dose on the SAME failure — the teto6/energia2 lesson
//! recorded in `ESCADA_DIFICULDADE.md` (right dose window: mortality
//! 5-20%, no more). Positive-only pressure keeps the incentive (chasing the ghost
//! pays) without risking blowing the mortality window.
//!
//! ## Sectors the ghost never reached
//!
//! The previous champion may have died/stalled before crossing all sectors.
//! For a sector it never reached, comparing time makes no sense (there is no T
//! to compare against) — in that case the being that gets there is, by definition,
//! going beyond where the best of the past generation got. We treat this as a
//! FIXED advantage (`VANTAGEM_SETOR_NAO_ALCANCADO`, +5.0 s), large enough to
//! saturate `bonus_vida` at the cap (it does not need to be exact — it only needs to
//! be "clearly ahead" without relying on a real infinity reaching later arithmetic).
//!
//! ## Life cycle (same process, no disk)
//!
//! `Fantasma` is `Clone`; the intent is: at the end of generation N, the orchestrator
//! calls `Fantasma::gravar(...)` from the best being's crossings and keeps the value
//! (e.g. `Option<Fantasma>` in a variable of the generation loop); in generation N+1,
//! each tick consults that same value via `vantagem()`/`bonus_vida()`. It does not
//! cross processes, has no byte serialization — it is pure RAM between generations.

/// Number of circuit sectors. Mirrors `N_CHECKPOINTS` from `statics.rs` (24).
///
/// Duplicated here on purpose: this file has exclusive ownership and compiles on its own
/// (`rustc --edition 2021 --crate-type lib`, `std` only), so it cannot depend on
/// `crate::statics`. If `N_CHECKPOINTS` changes from 24, this value must follow —
/// documented for whoever wires the two modules together.
pub const N_SETORES: u32 = 24;

/// Physics ticks per second (64 Hz, project constant — see `context.rs`,
/// `statics.rs`: 1,920 ticks = 30 s at 64 Hz).
const TICKS_POR_SEGUNDO: f32 = 64.0;

/// Advantage assigned when the requested sector is beyond where the ghost got
/// (or outside the array). Fixed and finite on purpose: there is no need to know "how
/// far ahead", only that it is a large advantage — `bonus_vida` saturates well before it.
pub const VANTAGEM_SETOR_NAO_ALCANCADO: f32 = 5.0;

/// Advantage (in seconds) from which `bonus_vida` is already at the cap.
const VANTAGEM_CAP_S: f32 = 2.0;

/// Cap of extra life granted per tick when the advantage is ≥ `VANTAGEM_CAP_S`.
/// Same unit as `life_earn`/`wall_life` in `main.rs`: life ticks per simulation
/// tick (here it is always double the default of earned-life per advance; it is
/// intentionally small next to the initial breath of 1,920 ticks/30s).
pub const BONUS_VIDA_CAP: f32 = 2.0;

/// Time (in ticks) the previous generation's CHAMPION took to reach each of the
/// `N_SETORES` circuit sectors, from the start line (tick 0).
///
/// `tempos_setor[s] == f32::INFINITY` means "the champion never reached sector
/// `s` in this life" (it died, got stuck, or the recorded crossing list did not
/// cover that sector).
#[derive(Clone, Debug, Default)]
pub struct Fantasma {
    pub tempos_setor: Vec<f32>,
}

impl Fantasma {
    /// Builds the ghost from the sector crossings of the best being of the
    /// generation that just ended: each item is `(sector, tick)`, the tick at which
    /// the being entered that sector for the first time.
    ///
    /// The list may be:
    /// - **incomplete**: sectors never crossed simply do not appear, and stay
    ///   `f32::INFINITY` in `tempos_setor`.
    /// - **unordered or with repetition** (e.g. the same sector index crossed again
    ///   on a following lap, since `checkpoint` in `main.rs` counts modulo
    ///   `N_CHECKPOINTS`): we keep the FIRST crossing (lowest tick) of each
    ///   sector — it is what defines "when the champion passed there" on the first
    ///   lap, which is the reference that matters for sector-by-sector pressure.
    ///
    /// Indices `setor >= N_SETORES` are ignored (defensive — it should not
    /// happen if the caller already applies `% N_CHECKPOINTS`, but it is no reason
    /// to panic).
    pub fn gravar(cruzamentos: &[(u32, u32)]) -> Fantasma {
        let mut tempos_setor = vec![f32::INFINITY; N_SETORES as usize];
        for &(setor, tick) in cruzamentos {
            if let Some(slot) = tempos_setor.get_mut(setor as usize) {
                let tick_f = tick as f32;
                if tick_f < *slot {
                    *slot = tick_f;
                }
            }
        }
        Fantasma { tempos_setor }
    }

    /// The being's advantage over the ghost at `setor_atual`, in SECONDS.
    ///
    /// Positive = the being is ahead (arrived earlier than the ghost arrived at
    /// that sector). Negative = it is behind. `vantagem = (T − tick_atual) / 64`,
    /// where `T` is the tick at which the ghost reached `setor_atual`.
    ///
    /// If the ghost never reached `setor_atual` (or the index is outside what was
    /// recorded), returns the fixed advantage `VANTAGEM_SETOR_NAO_ALCANCADO` — see
    /// the module comment.
    pub fn vantagem(&self, setor_atual: u32, tick_atual: u32) -> f32 {
        let t_fantasma = self
            .tempos_setor
            .get(setor_atual as usize)
            .copied()
            .unwrap_or(f32::INFINITY);
        if !t_fantasma.is_finite() {
            return VANTAGEM_SETOR_NAO_ALCANCADO;
        }
        (t_fantasma - tick_atual as f32) / TICKS_POR_SEGUNDO
    }

    /// Life bonus (in ticks/tick, same unit as `life_earn`) for a given
    /// advantage in seconds (typically the output of `vantagem()`).
    ///
    /// Asymmetric on purpose (see the module comment — cross currency): ahead of
    /// the ghost GAINS extra life, linear from 0 to `VANTAGEM_CAP_S` (2 s) of
    /// advantage, saturating at `BONUS_VIDA_CAP` (2 life/tick) from there on. Behind
    /// the ghost (advantage ≤ 0) it does NOT lose life — returns 0.0. Punishing
    /// whoever is behind is already the job of normal earned-life per advance;
    /// duplicating the punishment in the same currency (life) for the same cause
    /// (slowness) is a double dose, not a new dose.
    pub fn bonus_vida(&self, vantagem_s: f32) -> f32 {
        (vantagem_s.max(0.0) / VANTAGEM_CAP_S).min(1.0) * BONUS_VIDA_CAP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ghost that crossed the `N_SETORES` sectors at a constant pace of 100 ticks
    /// per sector (sector 0 at tick 0, sector 1 at tick 100, ...).
    fn fantasma_constante() -> Fantasma {
        let cruzamentos: Vec<(u32, u32)> = (0..N_SETORES).map(|s| (s, s * 100)).collect();
        Fantasma::gravar(&cruzamentos)
    }

    #[test]
    fn gravar_fantasma_completo_preenche_todos_os_setores_finitos() {
        let f = fantasma_constante();
        assert_eq!(f.tempos_setor.len(), N_SETORES as usize);
        assert!(f.tempos_setor.iter().all(|t| t.is_finite()));
        assert_eq!(f.tempos_setor[0], 0.0);
        assert_eq!(f.tempos_setor[23], 2300.0);
    }

    #[test]
    fn ser_a_frente_do_fantasma_completo_ganha_vantagem_e_bonus_positivos() {
        let f = fantasma_constante();
        // Ghost reached sector 5 at tick 500 (5*100); being arrives at tick 436,
        // 64 ticks (1 s) earlier.
        let vant = f.vantagem(5, 436);
        assert!(
            (vant - 1.0).abs() < 1e-4,
            "expected advantage ~1.0s, got {vant}"
        );
        let bonus = f.bonus_vida(vant);
        assert!(
            (bonus - 1.0).abs() < 1e-4,
            "expected bonus ~1.0, got {bonus}"
        );
    }

    #[test]
    fn ser_atras_do_fantasma_completo_tem_vantagem_negativa_e_bonus_zero() {
        let f = fantasma_constante();
        // Ghost reached sector 10 at tick 1000; being arrives at 1128 (2s late).
        let vant = f.vantagem(10, 1128);
        assert!(
            (vant + 2.0).abs() < 1e-4,
            "expected advantage ~-2.0s, got {vant}"
        );
        assert_eq!(
            f.bonus_vida(vant),
            0.0,
            "behind loses nothing, but gains nothing either"
        );
    }

    #[test]
    fn fantasma_incompleto_marca_setores_nao_cruzados_como_infinito() {
        // Ghost died/stalled before leaving sector 11 (only crossed 0..=11).
        let cruzamentos: Vec<(u32, u32)> = (0..=11u32).map(|s| (s, s * 80)).collect();
        let f = Fantasma::gravar(&cruzamentos);
        assert!(f.tempos_setor[11].is_finite());
        assert!(f.tempos_setor[12].is_infinite());
        assert!(f.tempos_setor[23].is_infinite());

        // A being reaching sector 15 (the ghost never got there) receives the
        // fixed advantage, not NaN nor a panic from dividing infinity.
        let vant = f.vantagem(15, 999_999);
        assert_eq!(vant, VANTAGEM_SETOR_NAO_ALCANCADO);
        assert_eq!(f.bonus_vida(vant), BONUS_VIDA_CAP);
    }

    #[test]
    fn setor_zero_na_largada_compara_corretamente() {
        // Ghost crosses sector 0 exactly at the start (tick 0) — common case,
        // since `checkpoint[i] == esperado == 0` usually holds from tick 0
        // (see main.rs: sector computed from position, expected starts at 0).
        let cruzamentos = vec![(0u32, 0u32), (1u32, 90u32)];
        let f = Fantasma::gravar(&cruzamentos);
        assert_eq!(f.tempos_setor[0], 0.0);

        // The being also starts at tick 0: exact tie, advantage 0, no bonus (but
        // no punishment either).
        assert_eq!(f.vantagem(0, 0), 0.0);
        assert_eq!(f.bonus_vida(0.0), 0.0);

        // The being takes 1s (64 ticks) longer to "leave" sector 0: 1s behind.
        let vant = f.vantagem(0, 64);
        assert!((vant + 1.0).abs() < 1e-4);
        assert_eq!(f.bonus_vida(vant), 0.0);
    }

    #[test]
    fn setor_zero_nunca_gravado_e_tratado_como_nao_alcancado() {
        // Crossing list that only starts at sector 1 — sector 0 never appears
        // explicitly (e.g. recording bug, or a track where the first recorded
        // event is already sector 1). Must not lock up: becomes "not reached".
        let cruzamentos = vec![(1u32, 90u32)];
        let f = Fantasma::gravar(&cruzamentos);
        assert!(f.tempos_setor[0].is_infinite());
        assert_eq!(f.vantagem(0, 500), VANTAGEM_SETOR_NAO_ALCANCADO);
    }

    #[test]
    fn cruzamento_duplicado_do_mesmo_setor_mantem_o_menor_tick() {
        // Simulates two entries for sector 3 (e.g. modulo across laps, or a list
        // out of order) — the FIRST crossing (lowest tick) is the one that counts.
        let cruzamentos = vec![(3u32, 300u32), (3u32, 50u32)];
        let f = Fantasma::gravar(&cruzamentos);
        assert_eq!(f.tempos_setor[3], 50.0);
    }

    #[test]
    fn setor_alem_do_indice_gravado_degrada_para_nao_alcancado() {
        // A caller forgetting the modulo (setor_atual >= N_SETORES) must not
        // cause an index panic — `vantagem` degrades gracefully.
        let f = Fantasma::gravar(&[(0, 0)]);
        let vant = f.vantagem(999, 10);
        assert_eq!(vant, VANTAGEM_SETOR_NAO_ALCANCADO);
    }

    #[test]
    fn bonus_vida_e_linear_ate_o_cap_e_depois_satura() {
        let f = Fantasma::default();
        assert_eq!(f.bonus_vida(0.0), 0.0);
        assert!((f.bonus_vida(1.0) - 1.0).abs() < 1e-4); // halfway -> half the cap
        assert_eq!(f.bonus_vida(2.0), BONUS_VIDA_CAP); // exactly at the cap
        assert_eq!(f.bonus_vida(10.0), BONUS_VIDA_CAP); // beyond the cap, still capped
        assert_eq!(f.bonus_vida(-3.0), 0.0); // behind loses nothing
    }

    #[test]
    fn fantasma_default_ou_vazio_da_vantagem_maxima_em_qualquer_setor() {
        // No previous ghost (e.g. generation 0, no one to compare against yet):
        // `Default` produces an empty `tempos_setor`, which behaves as "every
        // sector not reached" — every being receives the MAXIMUM bonus. Harmless:
        // the cap is small (+2 life/tick) and the real pressure switches on by
        // itself as soon as there is a true ghost in the following generation.
        let f = Fantasma::default();
        assert!(f.tempos_setor.is_empty());
        assert_eq!(f.vantagem(0, 0), VANTAGEM_SETOR_NAO_ALCANCADO);
        assert_eq!(f.bonus_vida(f.vantagem(0, 0)), BONUS_VIDA_CAP);
    }
}
