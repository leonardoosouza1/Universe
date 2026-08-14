//! Rotação determinística de pistas de treino (P1, docs/PLANO_EVOLUCAO.md §P1).
//!
//! Puro e sem GPU de propósito — `pista_da_gen`/`parse_pistas` não tocam wgpu nem
//! `VoxelWorld`, então são testáveis com `cargo test --bin gpu_evolve` sem hardware.
//!
//! # Por que "conjunto vazio = pista 9001 fixa"
//!
//! O campeão atual (99,25% sorteio inicial, ver PLANO_EVOLUCAO.md) treina SEMPRE na
//! mesma pista (seed 9001, `--same-start`) — é o A/B em andamento. `--pistas` tem que
//! ser estritamente aditivo: sem a flag, `pistas` chega vazio em `pista_da_gen`, que
//! devolve 9001 pra QUALQUER geração — bit-a-bit o mesmo hardcode que `main.rs` tinha
//! antes desta mudança (`VoxelWorld::new(9001, track_width)`, fora do laço).

/// Pista da geração `gen_abs`, por rotação determinística (`pistas[gen_abs % len]`).
///
/// Lista vazia devolve 9001 — a pista fixa que o binário sempre usou antes de
/// `--pistas` existir. Determinístico por construção: mesma `gen_abs` + mesmo
/// `pistas` sempre devolvem a mesma seed, o que é o que torna a bateria reproduzível
/// (dois processos com o mesmo `--pistas` treinam exatamente na mesma sequência de
/// pistas, gen a gen).
pub fn pista_da_gen(pistas: &[u64], gen_abs: usize) -> u64 {
    if pistas.is_empty() {
        9001
    } else {
        pistas[gen_abs % pistas.len()]
    }
}

/// Parseia `--pistas "9001,9002,9003"` em uma lista de seeds.
///
/// Vazio (flag ausente ou string vazia) devolve `Vec::new()` — é o valor que faz
/// `pista_da_gen` cair no braço de controle (pista 9001 fixa). Tokens que não
/// parseiam como `u64` são só ignorados (não abortam o parse inteiro): um `--pistas`
/// digitado errado deve falhar visivelmente por ter MENOS pistas que o esperado no
/// log de `[pista] gen ...`, não travar o binário inteiro por um dígito trocado.
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

    /// Sem `--pistas` (lista vazia), toda geração usa 9001 — o A/B em andamento
    /// (campeão treinado só na seed 9001, `--same-start`) não pode ser invalidado
    /// por esta mudança.
    #[test]
    fn conjunto_vazio_e_pista_fixa_9001_comportamento_atual() {
        // `gen` é palavra reservada nesta edição do Rust (sintaxe de geradores) —
        // por isso `g` em vez de `gen` nestes testes.
        for g in [0usize, 1, 5, 100, 4000] {
            assert_eq!(
                pista_da_gen(&[], g), 9001,
                "sem --pistas, tem que reproduzir bit-a-bit o binário de antes desta mudança"
            );
        }
    }

    /// A mesma geração tem que sortear sempre a mesma pista — é o que torna a
    /// bateria reproduzível entre processos/retomadas.
    #[test]
    fn rotacao_e_deterministica_mesma_gen_mesma_pista() {
        let pistas = vec![9001, 123, 64, 256, 512, 768, 1024, 31337];
        for g in [0usize, 4, 7, 40, 4000] {
            let a = pista_da_gen(&pistas, g);
            let b = pista_da_gen(&pistas, g);
            assert_eq!(a, b, "gen {g}: duas chamadas com o mesmo conjunto divergiram");
        }
    }

    /// Percorre o conjunto em ordem e repete ao voltar ao começo — `gen % len`.
    #[test]
    fn rotacao_percorre_o_conjunto_em_ordem_e_repete() {
        let pistas = vec![9001, 123, 64];
        assert_eq!(pista_da_gen(&pistas, 0), 9001);
        assert_eq!(pista_da_gen(&pistas, 1), 123);
        assert_eq!(pista_da_gen(&pistas, 2), 64);
        assert_eq!(pista_da_gen(&pistas, 3), 9001, "gen 3 tem que voltar ao início do conjunto");
        assert_eq!(pista_da_gen(&pistas, 4), 123);
        assert_eq!(pista_da_gen(&pistas, 8), pistas[8 % pistas.len()]);
    }

    /// Lista de uma pista só se comporta como pista fixa (qualquer seed, não só
    /// 9001) — cobre `--pistas 42` como equivalente de "fixa em 42".
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
        // "abc" não parseia como u64 — é descartado, os outros dois sobrevivem.
        assert_eq!(parse_pistas("9001,abc,123"), vec![9001, 123]);
    }

    #[test]
    fn parse_pistas_trailing_comma_nao_gera_zero_espurio() {
        assert_eq!(parse_pistas("9001,123,"), vec![9001, 123]);
    }
}
