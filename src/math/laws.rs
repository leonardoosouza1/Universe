/// Leis derivadas do ecossistema IARA — custo de busca e metabolismo.
///
/// Documentadas em MATH_INVENTORY.md (F27, F28) e confirmadas empiricamente
/// nos projetos iara-agua e Universe.
///
/// - `search_noise_threshold` (Lei do Garimpo): limiar esperado do melhor de M amostras
///   de ruído puro; um ganho de fitness só é "real" se exceder este limiar.
/// - `aic_cost` (AIC/metabolismo): penalidade de Akaike baseada na variância residual.
/// - `ric_penalty` (RIC): variante bayesiana da Lei do Garimpo usada em poda de rede.

/// Limiar do Garimpo: ganho esperado do melhor de M amostras de ruído N(0, σ²).
///
/// Fórmula: `σ · √(2 · ln M)`
///
/// Interpretação: se buscarmos o melhor de M candidatos gerados por puro ruído
/// gaussiano N(0, σ²), o valor esperado do máximo é `σ·√(2·ln M)`. Um ganho
/// de fitness só supera o teto de busca estocástica se exceder este limiar.
///
/// Retorna 0.0 se `sigma <= 0` ou `m_alternatives <= 1`.
///
/// Fonte canônica desta função (E1 extrai de brain_capacity.rs).
pub fn search_noise_threshold(sigma: f32, m_alternatives: usize) -> f32 {
    if sigma <= 0.0 || m_alternatives <= 1 {
        return 0.0;
    }
    sigma * (2.0 * (m_alternatives as f32).ln()).sqrt()
}

/// Custo AIC — critério de informação de Akaike simplificado.
///
/// Fórmula: `AIC ≈ n_samples · ln(residual_variance) + 2 · n_params`
///
/// A minimização de AIC penaliza modelos com mais parâmetros que não reduzem
/// suficientemente a variância residual. Usado como "metabolismo" na lei
/// `met_c = σ²(1 + 2·dof/N)` do projeto iara-agua.
///
/// Esta função retorna o custo por amostra (normalizado): `ln(σ²) + 2k/N`
/// onde `k = n_params` e `N = n_samples`.
///
/// Retorna 0.0 se `residual_variance <= 0` ou `n_samples == 0`.
pub fn aic_cost(residual_variance: f32, n_params: usize, n_samples: usize) -> f32 {
    if residual_variance <= 0.0 || n_samples == 0 {
        return 0.0;
    }
    residual_variance.ln() + 2.0 * n_params as f32 / n_samples as f32
}

/// Penalidade RIC — variante Bayesiana da Lei do Garimpo.
///
/// Fórmula: `σ · √(2 · ln M)` (mesma que search_noise_threshold).
///
/// O RIC (Risk Inflation Criterion) penaliza a seleção de variáveis/neurônios
/// pelo custo esperado de escolher o melhor de M alternativas aleatórias. É a
/// generalização do Deflated Sharpe Ratio para seleção de neurônios.
///
/// Na poda de rede: um neurônio só justifica existir se seu ganho > ric_penalty.
///
/// Idêntica à fórmula de `search_noise_threshold` — separada como alias semântico
/// para deixar claro o contexto de poda vs. busca evolucionária.
pub fn ric_penalty(sigma: f32, m_alternatives: usize) -> f32 {
    search_noise_threshold(sigma, m_alternatives)
}

// ---------------------------------------------------------------------------
// Testes unitários
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garimpo_monotono_com_m() {
        let t10 = search_noise_threshold(1.0, 10);
        let t100 = search_noise_threshold(1.0, 100);
        let t1000 = search_noise_threshold(1.0, 1000);
        assert!(t10 < t100, "limiar deve crescer: {} < {}", t10, t100);
        assert!(t100 < t1000, "limiar deve crescer: {} < {}", t100, t1000);
    }

    #[test]
    fn garimpo_zero_quando_sigma_zero() {
        assert_eq!(search_noise_threshold(0.0, 1000), 0.0);
    }

    #[test]
    fn garimpo_zero_quando_m_um() {
        assert_eq!(search_noise_threshold(1.5, 1), 0.0);
    }

    #[test]
    fn garimpo_valor_numerico_conhecido() {
        // σ=1, M=2 → √(2·ln2) ≈ 1.1774
        let expected = (2.0_f32 * 2.0_f32.ln()).sqrt();
        let got = search_noise_threshold(1.0, 2);
        assert!(
            (got - expected).abs() < 1e-5,
            "σ=1 M=2: esperado {}, obteve {}",
            expected,
            got
        );
    }

    #[test]
    fn garimpo_escala_linear_com_sigma() {
        // Limiar deve ser proporcional a σ
        let t1 = search_noise_threshold(1.0, 100);
        let t2 = search_noise_threshold(2.0, 100);
        assert!(
            (t2 / t1 - 2.0).abs() < 1e-5,
            "limiar deve escalar 2× com 2σ"
        );
    }

    #[test]
    fn aic_custo_cresce_com_params() {
        // Mesma variância residual, mais parâmetros → AIC maior
        let aic_simples = aic_cost(0.1, 5, 100);
        let aic_complexo = aic_cost(0.1, 50, 100);
        assert!(
            aic_complexo > aic_simples,
            "AIC deve crescer com mais parâmetros"
        );
    }

    #[test]
    fn aic_zero_quando_variancia_zero() {
        assert_eq!(aic_cost(0.0, 5, 100), 0.0);
    }

    #[test]
    fn aic_zero_quando_n_samples_zero() {
        assert_eq!(aic_cost(0.5, 5, 0), 0.0);
    }

    #[test]
    fn ric_identico_ao_garimpo() {
        // RIC é alias semântico de search_noise_threshold
        let sigma = 1.5_f32;
        let m = 50usize;
        assert_eq!(ric_penalty(sigma, m), search_noise_threshold(sigma, m));
    }
}
