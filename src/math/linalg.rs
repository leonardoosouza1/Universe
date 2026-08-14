/// Álgebra linear pura — sem dependências de GPU ou Bevy.
///
/// Funções documentadas em REFACTOR_PLAN.md §E1 e MATH_INVENTORY.md §1.5 (F27):
/// - `participation_ratio`: extraído/reimplementado de `src/brain_capacity.rs`
/// - `spectral_radius`: raio espectral de W_rec via power-iteration
/// - `effective_rank`: rank efetivo de valores singulares
///
/// NOTA: `src/brain_capacity.rs` NÃO é removido neste estágio (E1). A função aqui
/// é a fonte canônica; brain_capacity.rs re-exportará daqui no E2.

// ---------------------------------------------------------------------------
// Autovalores por Jacobi cíclico (movida de brain_capacity.rs)
// ---------------------------------------------------------------------------

/// Retorna os N autovalores de uma matriz simétrica NxN (linha-maior).
///
/// Usa Jacobi cíclico: gira todos os pares (i,j) com i<j em sequência,
/// repetindo até que a norma fora-da-diagonal seja < tol ou max_iter seja atingido.
/// Para matrizes pequenas (<= 32×32) converge em dezenas de sweeps.
///
/// Privado: chamadores externos usam `participation_ratio` ou `effective_rank`.
fn jacobi_eigenvalues(mat: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(mat.len(), n * n, "matriz deve ser n×n");
    let mut a: Vec<f32> = mat.to_vec();
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
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                let dpp = t * apq;
                a[p * n + p] -= dpp;
                a[q * n + q] += dpp;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

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

    (0..n).map(|i| a[i * n + i]).collect()
}

/// Autovalores de uma matriz SIMÉTRICA n×n (linha-maior), via Jacobi cíclico.
///
/// Wrapper público de `jacobi_eigenvalues` (privada nesta seção). Existe porque
/// `analysis::activation` (E4) precisa do ESPECTRO INTEIRO de valores singulares de
/// W_L1 para `effective_rank` — `spectral_radius` (power-iteration) só devolve o
/// maior autovalor, que basta para o alarme de instabilidade mas não para rank
/// efetivo (que precisa da energia acumulada por todos os valores).
///
/// Não normaliza nem ordena — quem chama decide (ex.: `effective_rank` espera
/// ordem decrescente).
pub fn eigenvalues_symmetric(mat: &[f32], n: usize) -> Vec<f32> {
    jacobi_eigenvalues(mat, n)
}

// ---------------------------------------------------------------------------
// API pública
// ---------------------------------------------------------------------------

/// Calcula o Participation Ratio (PR) das ativações de uma camada.
///
/// `activations[t][j]` = ativação do neurônio `j` no passo de tempo `t`.
/// Centraliza cada coluna, computa covariância empírica (divisor T), extrai
/// autovalores via Jacobi, devolve:
///
/// ```text
/// PR = (Σ λ_i)² / Σ λ_i²
/// ```
///
/// PR ≈ N → todas as dimensões usadas (saturado).
/// PR ≈ 1 → apenas uma dimensão efetiva.
///
/// Retorna `None` se há amostras insuficientes (T ≤ width) ou width == 0.
///
/// Esta é a fonte canônica (REFACTOR_PLAN.md §E1). `brain_capacity::participation_ratio`
/// re-exportará daqui após E2.
pub fn participation_ratio(activations: &[Vec<f32>]) -> Option<f32> {
    let t = activations.len();
    if t == 0 {
        return None;
    }
    let width = activations[0].len();
    if width == 0 || t <= width {
        return None;
    }

    // Médias por coluna
    let mut col_mean = vec![0.0f32; width];
    for row in activations.iter() {
        for (j, &v) in row.iter().enumerate() {
            col_mean[j] += v;
        }
    }
    let inv_t = 1.0 / t as f32;
    for m in col_mean.iter_mut() {
        *m *= inv_t;
    }

    // Matriz de covariância empírica (divisor T, não T-1) — garante PR = width
    // quando todas as colunas são iid com variância igual.
    let mut cov = vec![0.0f32; width * width];
    for row in activations.iter() {
        for i in 0..width {
            let xi = row[i] - col_mean[i];
            for j in 0..width {
                cov[i * width + j] += xi * (row[j] - col_mean[j]);
            }
        }
    }
    for v in cov.iter_mut() {
        *v *= inv_t;
    }

    let eigenvalues = jacobi_eigenvalues(&cov, width);

    // Autovalores negativos por ruído numérico: tratar como zero
    let lambdas: Vec<f32> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();

    let sum1: f32 = lambdas.iter().sum();
    let sum2: f32 = lambdas.iter().map(|v| v * v).sum();

    if sum2 < 1e-30 {
        // Ativações constantes — posto zero; PR = 1 por convenção
        return Some(1.0);
    }

    Some(sum1 * sum1 / sum2)
}

/// Raio espectral de uma matriz quadrada N×N (maior valor absoluto de autovalor)
/// via power-iteration.
///
/// `w_rec` é a matriz linha-maior de dimensão `n × n`.
/// `max_iter` controla a convergência (100 é suficiente para matrizes < 200×200).
/// Retorna 0.0 se `n == 0` ou se a matriz é toda zeros.
///
/// Usado em `analysis/activation.rs` (E4) para detectar instabilidade recorrente
/// (raio > 1.0 → dinâmica potencialmente explosiva).
pub fn spectral_radius(w_rec: &[f32], n: usize, max_iter: usize) -> f32 {
    if n == 0 || w_rec.len() != n * n {
        return 0.0;
    }

    // Inicializa com vetor de uns normalizado
    let mut v: Vec<f32> = vec![1.0 / (n as f32).sqrt(); n];
    let mut eigenvalue = 0.0f32;

    for _ in 0..max_iter {
        // w = W_rec × v (produto matriz-vetor)
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += w_rec[i * n + j] * v[j];
            }
            w[i] = acc;
        }
        // Norma do resultado
        let norm: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm < 1e-30 {
            return 0.0; // matriz nula ou ortogonal ao vetor
        }
        eigenvalue = norm; // estimativa do maior autovalor em módulo
        // Normaliza para próxima iteração
        for (vi, &wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }

    eigenvalue
}

/// Gram menor de uma matriz retangular linha-maior `[rows × cols]`: `W · Wᵀ`
/// (`rows × rows`). Escolhido (em vez de `Wᵀ·W`, `cols × cols`) porque em L1
/// `rows=96 < cols=192` — Gram menor custa `O(rows² · cols)` em vez de `O(cols² · rows)`,
/// e o Jacobi que roda em cima é `O(rows³)` por sweep.
///
/// Os autovalores de `W·Wᵀ` são os valores singulares de `W` ao quadrado — mesma
/// informação que `Wᵀ·W`, metade dos elementos a computar quando `rows < cols`.
///
/// Movida de `analysis::activation` (E4) para cá em P0.5: passou a ter um segundo
/// chamador (`gpu::evolution::Genome::random`, inicialização espectral) e duplicar a
/// mesma conta em dois módulos é exatamente o tipo de duplicidade que diverge em
/// silêncio quando um lado muda e o outro não.
pub fn gram_matrix(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; rows * rows];
    for i in 0..rows {
        let ri = i * cols;
        for k in i..rows {
            let rk = k * cols;
            let mut acc = 0.0f32;
            for j in 0..cols {
                acc += w[ri + j] * w[rk + j];
            }
            g[i * rows + k] = acc;
            g[k * rows + i] = acc;
        }
    }
    g
}

/// Maior valor singular de uma matriz retangular `w` (`rows × cols`, linha-maior):
/// `sqrt(λ_max(W·Wᵀ))`, via [`gram_matrix`] + [`spectral_radius`] (power-iteration).
///
/// É o mesmo número que `analysis::activation::spectral_summary` reporta como
/// `LayerStats::spectral_radius` para W_L1 — aqui exposto como função pura pra poder
/// ser medido e corrigido NA INICIALIZAÇÃO do genoma (`Genome::random`, P0.5), antes
/// do primeiro tick, em vez de só diagnosticado depois do fato.
///
/// Retorna 0.0 se `rows == 0`, `cols == 0` ou `w.len() != rows * cols`.
pub fn largest_singular_value(w: &[f32], rows: usize, cols: usize) -> f32 {
    if rows == 0 || cols == 0 || w.len() != rows * cols {
        return 0.0;
    }
    let gram = gram_matrix(w, rows, cols);
    spectral_radius(&gram, rows, 100).max(0.0).sqrt()
}

/// Rank efetivo de um vetor de valores singulares (decrescente).
///
/// Define rank efetivo como o menor k tal que:
/// `Σ_{i=0}^{k-1} σ_i² / Σ_i σ_i² >= threshold`
///
/// `threshold = 0.95` captura 95% da energia espectral.
///
/// Retorna 0 se o vetor estiver vazio ou se toda a energia é zero.
pub fn effective_rank(singular_vals: &[f32], threshold: f32) -> usize {
    if singular_vals.is_empty() {
        return 0;
    }
    let total_energy: f32 = singular_vals.iter().map(|&s| s * s).sum();
    if total_energy < 1e-30 {
        return 0;
    }
    let mut cumulative = 0.0f32;
    for (k, &s) in singular_vals.iter().enumerate() {
        cumulative += s * s;
        if cumulative / total_energy >= threshold {
            return k + 1;
        }
    }
    singular_vals.len()
}

// ---------------------------------------------------------------------------
// Testes unitários
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Vetor uniforme: todas dimensões com variância igual → PR = N.
    #[test]
    fn pr_vetor_uniforme_e_n() {
        let n = 4usize;
        let t = 200usize;
        // Sinais ortogonais de amplitude igual → autovalores iguais → PR = N
        let freqs = [1.0_f32, 2.0, 3.0, 5.0];
        let acts: Vec<Vec<f32>> = (0..t)
            .map(|i| freqs.iter()
                .map(|&f| (2.0 * std::f32::consts::PI * f * i as f32 / t as f32).sin())
                .collect())
            .collect();
        let pr = participation_ratio(&acts).expect("PR deve existir");
        assert!(
            (pr - n as f32).abs() < 0.5,
            "PR de sinais ortogonais deve ser ≈ {}, obteve {}", n, pr
        );
    }

    /// Posto 1: todas as colunas proporcionais → PR ≈ 1.
    #[test]
    fn pr_posto_1_retorna_1() {
        let t = 100usize;
        let w = 4usize;
        let acts: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let base = (i as f32 / (t - 1) as f32) * 2.0 - 1.0;
                (0..w).map(|j| base * (j + 1) as f32).collect()
            })
            .collect();
        let pr = participation_ratio(&acts).expect("PR deve existir");
        assert!((pr - 1.0).abs() < 0.05, "PR posto-1 deve ser ≈ 1, obteve {}", pr);
    }

    /// Amostras insuficientes → None.
    #[test]
    fn pr_insuficiente_retorna_none() {
        let acts: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // T=2 <= width=2
        assert!(participation_ratio(&acts).is_none());
    }

    /// Raio espectral de matriz identidade = 1.0.
    #[test]
    fn raio_espectral_identidade_e_1() {
        let n = 4usize;
        let mut m = vec![0.0_f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        let r = spectral_radius(&m, n, 100);
        assert!((r - 1.0).abs() < 0.01, "raio espectral de I deve ser 1, obteve {}", r);
    }

    /// Raio espectral de matriz zero = 0.0.
    #[test]
    fn raio_espectral_zero_e_0() {
        let n = 3usize;
        let m = vec![0.0_f32; n * n];
        let r = spectral_radius(&m, n, 50);
        assert!(r < 1e-10, "raio espectral de zeros deve ser 0, obteve {}", r);
    }

    /// Raio espectral de escalar 2 = 2.0.
    #[test]
    fn raio_espectral_escalar_2() {
        // Matriz [[2.0]]: autovalor = 2.0
        let r = spectral_radius(&[2.0], 1, 10);
        assert!((r - 2.0).abs() < 0.01, "raio espectral de [[2]] deve ser 2, obteve {}", r);
    }

    /// Rank efetivo de vetor [1,0,0,...]: apenas 1 singular value → rank = 1.
    #[test]
    fn effective_rank_dominante() {
        let sv = [1.0_f32, 0.001, 0.001, 0.001];
        let r = effective_rank(&sv, 0.95);
        // σ₀² = 1.0, total ≈ 1.0 → rank = 1 cobre ≥ 95%
        assert_eq!(r, 1, "rank efetivo com valor dominante deve ser 1");
    }

    /// Rank efetivo de vetor uniforme: todos contribuem → rank = N.
    #[test]
    fn effective_rank_uniforme_e_n() {
        let sv = [1.0_f32, 1.0, 1.0, 1.0];
        let r = effective_rank(&sv, 0.95);
        // Energia acumulada: 25%, 50%, 75%, 100% → rank = 4
        assert_eq!(r, 4, "rank efetivo de σ uniformes deve ser N");
    }

    /// Rank efetivo de vetor vazio = 0.
    #[test]
    fn effective_rank_vazio() {
        assert_eq!(effective_rank(&[], 0.95), 0);
    }

    /// Autovalores de uma matriz diagonal são os próprios elementos da diagonal.
    #[test]
    fn eigenvalues_symmetric_diagonal() {
        let n = 3usize;
        let mut m = vec![0.0f32; n * n];
        let diag = [4.0f32, 1.0, 9.0];
        for i in 0..n { m[i * n + i] = diag[i]; }
        let mut eig = eigenvalues_symmetric(&m, n);
        eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut esperado = diag.to_vec();
        esperado.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in eig.iter().zip(esperado.iter()) {
            assert!((a - b).abs() < 1e-3, "autovalor {} != esperado {}", a, b);
        }
    }

    /// `largest_singular_value` de uma matriz ortonormal (identidade 4×4) = 1.0.
    #[test]
    fn largest_singular_value_identidade_e_1() {
        let n = 4usize;
        let mut w = vec![0.0f32; n * n];
        for i in 0..n { w[i * n + i] = 1.0; }
        let sv = largest_singular_value(&w, n, n);
        assert!((sv - 1.0).abs() < 0.05, "esperado ~1.0, obteve {sv}");
    }

    /// Matriz retangular (2×4, rows < cols, o caso de L1 192→96 na prática):
    /// duas linhas ortogonais com normas diferentes → maior valor singular = maior norma.
    #[test]
    fn largest_singular_value_retangular() {
        let w = vec![
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
        ];
        let sv = largest_singular_value(&w, 2, 4);
        assert!((sv - 3.0).abs() < 0.05, "esperado ~3.0, obteve {sv}");
    }

    /// Escalar toda a matriz por `c` escala o maior valor singular pelo mesmo `c`
    /// (linear) — é exatamente a propriedade que a inicialização espectral do genoma
    /// (P0.5) usa para mirar o alvo: `fator = alvo / medido`.
    #[test]
    fn largest_singular_value_escala_linear_com_a_matriz() {
        let base = vec![0.6_f32, -0.2, 0.1, 0.4, -0.5, 0.3, 0.2, -0.1];
        let sv_base = largest_singular_value(&base, 2, 4);
        let escalado: Vec<f32> = base.iter().map(|&v| v * 2.5).collect();
        let sv_escalado = largest_singular_value(&escalado, 2, 4);
        assert!(
            (sv_escalado - sv_base * 2.5).abs() < 1e-3,
            "esperado {} (= {} × 2.5), obteve {}", sv_base * 2.5, sv_base, sv_escalado
        );
    }

    /// Forma errada (`w.len() != rows*cols`) devolve 0.0 em vez de entrar em pânico.
    #[test]
    fn largest_singular_value_forma_errada_retorna_zero() {
        assert_eq!(largest_singular_value(&[1.0, 2.0, 3.0], 2, 2), 0.0);
        assert_eq!(largest_singular_value(&[], 0, 5), 0.0);
    }

    /// A soma dos autovalores é o traço — checagem cruzada barata e independente
    /// do Jacobi convergir em ordem específica.
    #[test]
    fn eigenvalues_symmetric_soma_e_traco() {
        let n = 3usize;
        // Matriz simétrica arbitrária.
        let m = vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 4.0,
        ];
        let traco: f32 = (0..n).map(|i| m[i * n + i]).sum();
        let eig = eigenvalues_symmetric(&m, n);
        let soma: f32 = eig.iter().sum();
        assert!((soma - traco).abs() < 1e-2, "soma autovalores {} != traço {}", soma, traco);
    }
}
