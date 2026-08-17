/// Critério medido para decidir se um cérebro merece crescer.
///
/// Duas condições (ambas precisam ser verdadeiras para justificar crescimento):
///
/// (A) SATURAÇÃO — Participation Ratio (PR) das ativações da camada.
///     PR = (Σλ_i)² / Σλ_i²  onde λ_i são autovalores da covariância.
///     PR ≈ largura ⇒ todas as dimensões em uso (saturado, pode valer crescer).
///     PR << largura ⇒ dimensões ociosas (crescer é só espaço desperdiçado).
///
/// (B) GANHO REAL — Lei do Garimpo do ecossistema IARA:
///     Escolhendo o melhor de M alternativas de ruído puro, o máximo esperado é
///     σ·√(2·ln M). Um ganho de fitness só é real se exceder esse limiar.

/// Veredito de crescimento, com razão discriminada.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Crescer é justificado: capacidade saturada E ganho excede ruído de busca.
    Grow,
    /// Há dimensões ociosas — crescer desperdiça memória/computação.
    WastedSpace,
    /// Capacidade saturada mas o ganho não excede o limiar do Garimpo.
    GainNotReal,
    /// Amostras insuficientes para estimar a covariância com confiança
    /// (exige pelo menos `width + 1` amostras).
    InsufficientData,
}

// ---------------------------------------------------------------------------
// Jacobi cíclico para autovalores de matriz simétrica NxN (sem dependências).
// ---------------------------------------------------------------------------

/// Retorna os N autovalores de uma matriz simétrica NxN (linha-maior).
/// Usa Jacobi cíclico: gira todos os pares (i,j) i<j em sequência,
/// repetindo até que a norma fora-da-diagonal seja < tol ou max_iter seja atingido.
/// Para matrizes pequenas (<= 32×32) converge em dezenas de sweeps.
fn jacobi_eigenvalues(mat: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(mat.len(), n * n);
    let mut a: Vec<f32> = mat.to_vec(); // cópia mutável
    let tol = 1e-9_f32;
    let max_sweeps = 500;

    for _ in 0..max_sweeps {
        // Norma de Frobenius fora-da-diagonal ao quadrado
        let mut off = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                off += 2.0 * a[i * n + j] * a[i * n + j];
            }
        }
        if off < tol * tol {
            break;
        }

        // Um sweep: gira todos os pares (p, q) com p < q
        for p in 0..n {
            for q in (p + 1)..n {
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                // Ângulo de rotação de Jacobi
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                // Atualiza diagonal
                let dpp = t * apq;
                a[p * n + p] -= dpp;
                a[q * n + q] += dpp;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Atualiza linhas/colunas r != p, r != q
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let new_arp = arp - s * (arq + tau * arp);
                    let new_arq = arq + s * (arp - tau * arq);
                    a[r * n + p] = new_arp;
                    a[p * n + r] = new_arp;
                    a[r * n + q] = new_arq;
                    a[q * n + r] = new_arq;
                }
            }
        }
    }

    // Diagonal = autovalores
    (0..n).map(|i| a[i * n + i]).collect()
}

// ---------------------------------------------------------------------------
// API pública
// ---------------------------------------------------------------------------

/// Calcula o Participation Ratio das ativações de uma camada.
///
/// `activations`: linhas = amostras no tempo, colunas = unidades da camada.
/// Centraliza cada coluna, computa covariância empírica, extrai autovalores,
/// devolve PR = (Σλ_i)² / Σλ_i².
///
/// Retorna `None` se há amostras insuficientes (T <= width) ou width == 0.
pub fn participation_ratio(activations: &[Vec<f32>]) -> Option<f32> {
    let t = activations.len();
    if t == 0 {
        return None;
    }
    let width = activations[0].len();
    if width == 0 || t <= width {
        // Precisa de mais amostras do que dimensões para covariância bem definida
        return None;
    }

    // Médias por coluna
    let mut mean = vec![0.0f32; width];
    for row in activations.iter() {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v;
        }
    }
    let inv_t = 1.0 / t as f32;
    for m in mean.iter_mut() {
        *m *= inv_t;
    }

    // Matriz de covariância empírica (width × width), divisão por T (não T-1)
    // para PR ser exatamente `width` quando todas as colunas são iid igual variância.
    let mut cov = vec![0.0f32; width * width];
    for row in activations.iter() {
        for i in 0..width {
            let xi = row[i] - mean[i];
            for j in 0..width {
                let xj = row[j] - mean[j];
                cov[i * width + j] += xi * xj;
            }
        }
    }
    let inv_t2 = 1.0 / t as f32;
    for v in cov.iter_mut() {
        *v *= inv_t2;
    }

    let eigenvalues = jacobi_eigenvalues(&cov, width);

    // Apenas autovalores positivos (ruído numérico pode produzir levemente negativos)
    let lambdas: Vec<f32> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();

    let sum1: f32 = lambdas.iter().sum();
    let sum2: f32 = lambdas.iter().map(|v| v * v).sum();

    if sum2 < 1e-30 {
        // Todas as ativações constantes — posto zero, PR = 1 por convenção
        return Some(1.0);
    }

    Some(sum1 * sum1 / sum2)
}

/// Saturação = PR / largura da camada, em [0, 1].
/// 1.0 ⇒ todas as dimensões ativamente usadas (saturado).
/// 0.0 ⇒ posto efetivo próximo de 1 (uma dimensão domina tudo).
pub fn saturation(activations: &[Vec<f32>], width: usize) -> Option<f32> {
    if width == 0 {
        return None;
    }
    participation_ratio(activations).map(|pr| (pr / width as f32).min(1.0))
}

/// Limiar do Garimpo: ganho esperado do melhor de M amostras de ruído puro N(0, σ²).
/// Fórmula: σ · √(2 · ln M).
/// Se σ == 0 ou M <= 1, retorna 0.0 (nenhum ganho é "real" a checar).
pub fn search_noise_threshold(sigma: f32, m_alternatives: usize) -> f32 {
    if sigma <= 0.0 || m_alternatives <= 1 {
        return 0.0;
    }
    sigma * (2.0 * (m_alternatives as f32).ln()).sqrt()
}

/// Combina condição (A) saturação e condição (B) limiar do Garimpo num veredito único.
///
/// Parâmetros:
/// - `activations`: histórico de ativações da camada a avaliar.
/// - `width`: largura nominal da camada (número de unidades).
/// - `saturation_threshold`: PR/largura acima do qual dizemos que está saturada (ex: 0.85).
/// - `fitness_gain`: melhoria de fitness observada após as M avaliações.
/// - `fitness_sigma`: desvio padrão do fitness na população.
/// - `m_alternatives`: número total de variantes avaliadas (pop × gerações).
pub fn growth_verdict(
    activations: &[Vec<f32>],
    width: usize,
    saturation_threshold: f32,
    fitness_gain: f32,
    fitness_sigma: f32,
    m_alternatives: usize,
) -> Verdict {
    // Amostras insuficientes para estimar PR
    let sat = match saturation(activations, width) {
        Some(s) => s,
        None => return Verdict::InsufficientData,
    };

    // (A) capacidade saturada?
    if sat < saturation_threshold {
        return Verdict::WastedSpace;
    }

    // (B) ganho excede limiar do Garimpo?
    let threshold = search_noise_threshold(fitness_sigma, m_alternatives);
    if fitness_gain <= threshold {
        return Verdict::GainNotReal;
    }

    Verdict::Grow
}

// ---------------------------------------------------------------------------
// Testes unitários com dados sintéticos de propriedade conhecida
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Constrói T amostras onde todas as colunas são múltiplos da coluna 0.
    /// Posto efetivo = 1 ⇒ PR deve ser ≈ 1.
    #[test]
    fn pr_rank1_matrix() {
        let t = 100usize;
        let width = 4usize;
        // Coluna j = (j+1) * coluna 0; coluna 0 varia de -1 a 1 linearmente.
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let base = (i as f32 / (t - 1) as f32) * 2.0 - 1.0; // [-1, 1]
                (0..width).map(|j| base * (j + 1) as f32).collect()
            })
            .collect();
        let pr = participation_ratio(&activations).expect("deve ter PR");
        // Posto 1 ⇒ PR = 1
        assert!(
            (pr - 1.0).abs() < 0.05,
            "PR de matriz posto-1 deve ser ≈ 1, obteve {}",
            pr
        );
    }

    /// k colunas independentes com variância igual ⇒ PR = k.
    #[test]
    fn pr_independent_equal_variance() {
        let t = 500usize;
        let width = 6usize;
        // Sinal determinístico ortogonal: coluna j = sin(2π·j·i/T)
        // Para T grande e frequências diferentes são aproximadamente ortogonais.
        // Usamos frequências primas para garantir ortogonalidade empírica.
        let freqs: [f32; 6] = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0];
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                freqs
                    .iter()
                    .map(|&f| (2.0 * std::f32::consts::PI * f * i as f32 / t as f32).sin())
                    .collect()
            })
            .collect();
        let pr = participation_ratio(&activations).expect("deve ter PR");
        // Senos ortogonais de mesma amplitude ⇒ PR ≈ width = 6
        assert!(
            (pr - width as f32).abs() < 0.5,
            "PR de sinais ortogonais de igual variância deve ser ≈ {}, obteve {}",
            width,
            pr
        );
    }

    /// Espectro decaindo (autovalores λ_i ∝ 1/i) ⇒ 1 < PR < k.
    #[test]
    fn pr_decaying_spectrum_between_1_and_k() {
        let t = 300usize;
        let width = 5usize;
        // Constrói ativações cujas colunas têm variâncias 1, 1/2, 1/3, 1/4, 1/5
        // e são ortogonais (frequências distintas).
        let freqs: [f32; 5] = [1.0, 2.0, 3.0, 5.0, 7.0];
        let scales: [f32; 5] = [
            1.0,
            0.5_f32.sqrt(),
            (1.0 / 3.0_f32).sqrt(),
            0.5,
            (1.0 / 5.0_f32).sqrt(),
        ];
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                freqs
                    .iter()
                    .zip(scales.iter())
                    .map(|(&f, &s)| {
                        s * (2.0 * std::f32::consts::PI * f * i as f32 / t as f32).sin()
                    })
                    .collect()
            })
            .collect();
        let pr = participation_ratio(&activations).expect("deve ter PR");
        assert!(
            pr > 1.0 && pr < width as f32,
            "PR com espectro decaindo deve estar entre 1 e {}, obteve {}",
            width,
            pr
        );
    }

    /// garimpo_threshold cresce estritamente com M.
    #[test]
    fn search_noise_grows_with_m() {
        let sigma = 1.0;
        let t10 = search_noise_threshold(sigma, 10);
        let t100 = search_noise_threshold(sigma, 100);
        let t1000 = search_noise_threshold(sigma, 1000);
        assert!(t10 < t100, "limiar deve crescer: {} < {}", t10, t100);
        assert!(t100 < t1000, "limiar deve crescer: {} < {}", t100, t1000);
    }

    /// garimpo_threshold = 0 quando σ = 0.
    #[test]
    fn search_noise_zero_when_sigma_zero() {
        assert_eq!(search_noise_threshold(0.0, 1000), 0.0);
    }

    /// garimpo_threshold = 0 quando M <= 1.
    #[test]
    fn search_noise_zero_when_m_one() {
        assert_eq!(search_noise_threshold(1.5, 1), 0.0);
    }

    /// InsufficientData quando há poucas amostras (T <= width).
    #[test]
    fn verdict_insufficient_data() {
        // width = 4, mas só 3 amostras
        let activations: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
        ];
        let v = growth_verdict(&activations, 4, 0.85, 1.0, 0.5, 100);
        assert_eq!(v, Verdict::InsufficientData);
    }

    /// WastedSpace quando PR está muito abaixo da largura.
    #[test]
    fn verdict_wasted_space() {
        // Posto 1 ⇒ saturação ≈ 1/4 = 0.25, abaixo de 0.85
        let t = 100usize;
        let width = 4usize;
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let base = i as f32 / t as f32;
                (0..width).map(|j| base * (j + 1) as f32).collect()
            })
            .collect();
        let v = growth_verdict(&activations, width, 0.85, 99.0, 0.5, 100);
        assert_eq!(v, Verdict::WastedSpace);
    }

    /// GainNotReal quando capacidade saturada mas ganho abaixo do limiar.
    #[test]
    fn verdict_gain_not_real() {
        // Sinais ortogonais ⇒ PR ≈ width (saturado)
        let t = 500usize;
        let width = 4usize;
        let freqs: [f32; 4] = [1.0, 2.0, 3.0, 5.0];
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                freqs
                    .iter()
                    .map(|&f| (2.0 * std::f32::consts::PI * f * i as f32 / t as f32).sin())
                    .collect()
            })
            .collect();
        // sigma=1, M=100 ⇒ limiar = √(2·ln100) ≈ 3.03; gain=1.0 < 3.03
        let v = growth_verdict(&activations, width, 0.85, 1.0, 1.0, 100);
        assert_eq!(v, Verdict::GainNotReal);
    }

    /// Grow quando saturado E ganho acima do limiar.
    #[test]
    fn verdict_grow() {
        let t = 500usize;
        let width = 4usize;
        let freqs: [f32; 4] = [1.0, 2.0, 3.0, 5.0];
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                freqs
                    .iter()
                    .map(|&f| (2.0 * std::f32::consts::PI * f * i as f32 / t as f32).sin())
                    .collect()
            })
            .collect();
        // sigma=1, M=100 ⇒ limiar ≈ 3.03; gain=10.0 > 3.03
        let v = growth_verdict(&activations, width, 0.85, 10.0, 1.0, 100);
        assert_eq!(v, Verdict::Grow);
    }

    /// Valor numérico conhecido do limiar do Garimpo.
    #[test]
    fn search_noise_threshold_known_value() {
        // σ=1, M=e ⇒ √(2·ln e) = √2 ≈ 1.4142
        let threshold = search_noise_threshold(1.0, std::f32::consts::E as usize);
        // E as usize = 2, ln(2) ≈ 0.693, √(2·0.693) ≈ 1.177
        let expected = (2.0 * (2.0_f32).ln()).sqrt();
        assert!(
            (threshold - expected).abs() < 1e-4,
            "esperado {}, obteve {}",
            expected,
            threshold
        );
    }

    /// saturation() retorna None quando width=0.
    #[test]
    fn saturation_none_on_zero_width() {
        let activations: Vec<Vec<f32>> = vec![vec![]];
        assert!(saturation(&activations, 0).is_none());
    }
}
