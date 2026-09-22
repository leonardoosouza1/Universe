//! Deterministic rotation of training tracks (P1, docs/PLANO_EVOLUCAO.md §P1).
//!
//! Pure and GPU-free on purpose — `pista_da_gen`/`parse_pistas` touch neither wgpu nor
//! `VoxelWorld`, so they are testable with `cargo test --bin gpu_evolve` without hardware.
//!
//! # Why "empty set = validated starter track 8005"
//!
//! The current champion (99.25% initial-draw, see PLANO_EVOLUCAO.md) trains ALWAYS on
//! Seed 9001 now fails the generator's geometric validation and falls through to an
//! explicitly unvalidated candidate. It is useful only for reproducing historical
//! checkpoints. A fresh population must start on a world whose road and geometry have
//! passed validation, so an omitted `--tracks` selects seed 8005.

/// Track for generation `gen_abs`, by deterministic rotation (`pistas[gen_abs % len]`).
///
/// An empty list returns 8005, the validated starter track. Deterministic by
/// construction: the same `gen_abs` + the same
/// `pistas` always return the same seed, which is what makes the battery reproducible
/// (two processes with the same `--tracks` train on exactly the same sequence of
/// tracks, generation by generation).
pub fn pista_da_gen(pistas: &[u64], gen_abs: usize) -> u64 {
    if pistas.is_empty() {
        8005
    } else {
        pistas[gen_abs % pistas.len()]
    }
}

/// `WorldSpec` analogue of `pista_da_gen`, for `--world`/`--world-file`
/// (`universe::world_spec`). Same rotation rule: `specs[gen_abs % specs.len()]`.
/// Caller must check `!specs.is_empty()` first — `--world` composes with the track
/// rotation / `--switch-on-complete` machinery exactly like `--tracks`, but wins over
/// it when both are present (see `resolve_world`, `main.rs`).
pub fn spec_da_gen(
    specs: &[universe::world_spec::WorldSpec],
    gen_abs: usize,
) -> universe::world_spec::WorldSpec {
    specs[gen_abs % specs.len()]
}

/// Parses `--tracks "9001,9002,9003"` into a list of seeds.
///
/// Empty (flag absent or empty string) returns `Vec::new()` — the value that makes
/// `pista_da_gen` select the validated starter track. Tokens that do not
/// parse as `u64` are simply ignored (they do not abort the whole parse): a mistyped
/// `--tracks` should fail visibly by having FEWER tracks than expected in the
/// `[pista] gen ...` log, not lock up the whole binary over one wrong digit.
pub fn parse_pistas(s: &str) -> Vec<u64> {
    s.split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse::<u64>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without `--tracks` (empty list), every generation uses the validated starter.
    #[test]
    fn empty_set_uses_validated_starter_track_8005() {
        // `gen` is a reserved word in this Rust edition (generator syntax) —
        // hence `g` instead of `gen` in these tests.
        for g in [0usize, 1, 5, 100, 4000] {
            assert_eq!(
                pista_da_gen(&[], g),
                8005,
                "fresh training must never default to the unvalidated 9001 fallback"
            );
        }
    }

    /// The same generation has to draw the same track every time — that is what makes
    /// the battery reproducible across processes/resumes.
    #[test]
    fn rotacao_e_deterministica_mesma_gen_mesma_pista() {
        let pistas = vec![9001, 123, 64, 256, 512, 768, 1024, 31337];
        for g in [0usize, 4, 7, 40, 4000] {
            let a = pista_da_gen(&pistas, g);
            let b = pista_da_gen(&pistas, g);
            assert_eq!(a, b, "gen {g}: two calls with the same set diverged");
        }
    }

    /// Walks the set in order and repeats when it wraps back to the start — `gen % len`.
    #[test]
    fn rotacao_percorre_o_conjunto_em_ordem_e_repete() {
        let pistas = vec![9001, 123, 64];
        assert_eq!(pista_da_gen(&pistas, 0), 9001);
        assert_eq!(pista_da_gen(&pistas, 1), 123);
        assert_eq!(pista_da_gen(&pistas, 2), 64);
        assert_eq!(
            pista_da_gen(&pistas, 3),
            9001,
            "gen 3 has to wrap back to the start of the set"
        );
        assert_eq!(pista_da_gen(&pistas, 4), 123);
        assert_eq!(pista_da_gen(&pistas, 8), pistas[8 % pistas.len()]);
    }

    /// A single-track list behaves as a fixed track (any seed, not just 9001) —
    /// covers `--tracks 42` as the equivalent of "fixed at 42".
    #[test]
    fn lista_de_uma_pista_e_fixa_nessa_pista() {
        let pistas = vec![777u64];
        for g in [0usize, 1, 50, 999] {
            assert_eq!(pista_da_gen(&pistas, g), 777);
        }
    }

    #[test]
    fn parse_pistas_csv_basico() {
        assert_eq!(parse_pistas("9001,9002,9003"), vec![9001, 9002, 9003]);
    }

    #[test]
    fn parse_pistas_ignora_espacos() {
        assert_eq!(parse_pistas(" 9001 , 9002 ,  9003"), vec![9001, 9002, 9003]);
    }

    #[test]
    fn parse_pistas_string_vazia_da_lista_vazia() {
        assert_eq!(parse_pistas(""), Vec::<u64>::new());
    }

    #[test]
    fn parse_pistas_token_invalido_e_ignorado_nao_aborta() {
        // "abc" does not parse as u64 — it is discarded, the other two survive.
        assert_eq!(parse_pistas("9001,abc,123"), vec![9001, 123]);
    }

    #[test]
    fn parse_pistas_trailing_comma_nao_gera_zero_espurio() {
        assert_eq!(parse_pistas("9001,123,"), vec![9001, 123]);
    }

    #[test]
    fn spec_da_gen_rotates_like_pista_da_gen() {
        let specs = vec![
            universe::world_spec::parse("seed=8005").unwrap(),
            universe::world_spec::parse("seed=8100,radius=9").unwrap(),
        ];
        assert_eq!(spec_da_gen(&specs, 0).seed, 8005);
        assert_eq!(spec_da_gen(&specs, 1).seed, 8100);
        assert_eq!(
            spec_da_gen(&specs, 2).seed,
            8005,
            "gen 2 must wrap back to index 0"
        );
    }
}
