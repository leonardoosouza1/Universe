/// Pure linear algebra — no GPU or Bevy dependencies.
///
/// Functions documented in REFACTOR_PLAN.md §E1 and MATH_INVENTORY.md §1.5 (F27):
/// - `participation_ratio`: extracted/reimplemented from `src/brain_capacity.rs`
/// - `spectral_radius`: spectral radius of W_rec via power-iteration
/// - `effective_rank`: effective rank of singular values
///
/// NOTE: `src/brain_capacity.rs` is NOT removed at this stage (E1). The function here
/// is the canonical source; brain_capacity.rs will re-export from here in E2.

// ---------------------------------------------------------------------------
// Eigenvalues by cyclic Jacobi (moved from brain_capacity.rs)
// ---------------------------------------------------------------------------

/// Returns the N eigenvalues of a symmetric NxN matrix (row-major).
///
/// Uses cyclic Jacobi: rotates all pairs (i,j) with i<j in sequence,
/// repeating until the off-diagonal norm is < tol or max_iter is reached.
/// For small matrices (<= 32×32) it converges in dozens of sweeps.
///
/// Private: external callers use `participation_ratio` or `effective_rank`.
fn jacobi_eigenvalues(mat: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(mat.len(), n * n, "matrix must be n×n");
    let mut a: Vec<f32> = mat.to_vec();
    let tol = 1e-9_f32;
    let max_sweeps = 500;

    for _ in 0..max_sweeps {
        // Squared off-diagonal Frobenius norm
        let mut off = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                off += 2.0 * a[i * n + j] * a[i * n + j];
            }
        }
        if off < tol * tol {
            break;
        }

        // One sweep: rotates all pairs (p, q) with p < q
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

/// Eigenvalues of a SYMMETRIC n×n matrix (row-major), via cyclic Jacobi.
///
/// Public wrapper of `jacobi_eigenvalues` (private in this section). It exists because
/// `analysis::activation` (E4) needs the WHOLE SPECTRUM of singular values of
/// W_L1 for `effective_rank` — `spectral_radius` (power-iteration) only returns the
/// largest eigenvalue, which is enough for the instability alarm but not for effective
/// rank (which needs the energy accumulated by all values).
///
/// Neither normalizes nor sorts — the caller decides (e.g.: `effective_rank` expects
/// decreasing order).
pub fn eigenvalues_symmetric(mat: &[f32], n: usize) -> Vec<f32> {
    jacobi_eigenvalues(mat, n)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes the Participation Ratio (PR) of a layer's activations.
///
/// `activations[t][j]` = activation of neuron `j` at time step `t`.
/// Centers each column, computes the empirical covariance (divisor T), extracts
/// eigenvalues via Jacobi, returns:
///
/// ```text
/// PR = (Σ λ_i)² / Σ λ_i²
/// ```
///
/// PR ≈ N → all dimensions used (saturated).
/// PR ≈ 1 → only one effective dimension.
///
/// Returns `None` if there are insufficient samples (T ≤ width) or width == 0.
///
/// This is the canonical source (REFACTOR_PLAN.md §E1). `brain_capacity::participation_ratio`
/// will re-export from here after E2.
pub fn participation_ratio(activations: &[Vec<f32>]) -> Option<f32> {
    let t = activations.len();
    if t == 0 {
        return None;
    }
    let width = activations[0].len();
    if width == 0 || t <= width {
        return None;
    }

    // Per-column means
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

    // Empirical covariance matrix (divisor T, not T-1) — guarantees PR = width
    // when all columns are iid with equal variance.
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

    // Negative eigenvalues from numerical noise: treat as zero
    let lambdas: Vec<f32> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();

    let sum1: f32 = lambdas.iter().sum();
    let sum2: f32 = lambdas.iter().map(|v| v * v).sum();

    if sum2 < 1e-30 {
        // Constant activations — rank zero; PR = 1 by convention
        return Some(1.0);
    }

    Some(sum1 * sum1 / sum2)
}

/// Spectral radius of a square N×N matrix (largest absolute eigenvalue)
/// via power-iteration.
///
/// `w_rec` is the row-major matrix of dimension `n × n`.
/// `max_iter` controls convergence (100 is sufficient for matrices < 200×200).
/// Returns 0.0 if `n == 0` or if the matrix is all zeros.
///
/// Used in `analysis/activation.rs` (E4) to detect recurrent instability
/// (radius > 1.0 → potentially explosive dynamics).
pub fn spectral_radius(w_rec: &[f32], n: usize, max_iter: usize) -> f32 {
    if n == 0 || w_rec.len() != n * n {
        return 0.0;
    }

    // Initializes with a normalized vector of ones
    let mut v: Vec<f32> = vec![1.0 / (n as f32).sqrt(); n];
    let mut eigenvalue = 0.0f32;

    for _ in 0..max_iter {
        // w = W_rec × v (matrix-vector product)
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += w_rec[i * n + j] * v[j];
            }
            w[i] = acc;
        }
        // Norm of the result
        let norm: f32 = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm < 1e-30 {
            return 0.0; // null matrix or orthogonal to the vector
        }
        eigenvalue = norm; // estimate of the largest eigenvalue in modulus
        // Normalizes for the next iteration
        for (vi, &wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }

    eigenvalue
}

/// Smaller Gram of a row-major rectangular matrix `[rows × cols]`: `W · Wᵀ`
/// (`rows × rows`). Chosen (instead of `Wᵀ·W`, `cols × cols`) because in L1
/// `rows=96 < cols=192` — the smaller Gram costs `O(rows² · cols)` instead of `O(cols² · rows)`,
/// and the Jacobi that runs on top is `O(rows³)` per sweep.
///
/// The eigenvalues of `W·Wᵀ` are the singular values of `W` squared — the same
/// information as `Wᵀ·W`, half the elements to compute when `rows < cols`.
///
/// Moved here from `analysis::activation` (E4) in P0.5: it gained a second
/// caller (`gpu::evolution::Genome::random`, spectral initialization) and duplicating the
/// same computation in two modules is exactly the kind of duplication that diverges
/// silently when one side changes and the other does not.
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

/// Largest singular value of a rectangular matrix `w` (`rows × cols`, row-major):
/// `sqrt(λ_max(W·Wᵀ))`, via [`gram_matrix`] + [`spectral_radius`] (power-iteration).
///
/// It is the same number that `analysis::activation::spectral_summary` reports as
/// `LayerStats::spectral_radius` for W_L1 — exposed here as a pure function so it can
/// be measured and corrected AT GENOME INITIALIZATION (`Genome::random`, P0.5), before
/// the first tick, instead of only diagnosed after the fact.
///
/// Returns 0.0 if `rows == 0`, `cols == 0` or `w.len() != rows * cols`.
pub fn largest_singular_value(w: &[f32], rows: usize, cols: usize) -> f32 {
    if rows == 0 || cols == 0 || w.len() != rows * cols {
        return 0.0;
    }
    let gram = gram_matrix(w, rows, cols);
    spectral_radius(&gram, rows, 100).max(0.0).sqrt()
}

/// Effective rank of a vector of singular values (decreasing).
///
/// Defines effective rank as the smallest k such that:
/// `Σ_{i=0}^{k-1} σ_i² / Σ_i σ_i² >= threshold`
///
/// `threshold = 0.95` captures 95% of the spectral energy.
///
/// Returns 0 if the vector is empty or if all the energy is zero.
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
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform vector: all dimensions with equal variance → PR = N.
    #[test]
    fn pr_vetor_uniforme_e_n() {
        let n = 4usize;
        let t = 200usize;
        // Orthogonal signals of equal amplitude → equal eigenvalues → PR = N
        let freqs = [1.0_f32, 2.0, 3.0, 5.0];
        let acts: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                freqs
                    .iter()
                    .map(|&f| (2.0 * std::f32::consts::PI * f * i as f32 / t as f32).sin())
                    .collect()
            })
            .collect();
        let pr = participation_ratio(&acts).expect("PR must exist");
        assert!(
            (pr - n as f32).abs() < 0.5,
            "PR of orthogonal signals must be ≈ {}, got {}",
            n,
            pr
        );
    }

    /// Rank 1: all columns proportional → PR ≈ 1.
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
        let pr = participation_ratio(&acts).expect("PR must exist");
        assert!((pr - 1.0).abs() < 0.05, "rank-1 PR must be ≈ 1, got {}", pr);
    }

    /// Insufficient samples → None.
    #[test]
    fn pr_insuficiente_retorna_none() {
        let acts: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // T=2 <= width=2
        assert!(participation_ratio(&acts).is_none());
    }

    /// Spectral radius of the identity matrix = 1.0.
    #[test]
    fn raio_espectral_identidade_e_1() {
        let n = 4usize;
        let mut m = vec![0.0_f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        let r = spectral_radius(&m, n, 100);
        assert!(
            (r - 1.0).abs() < 0.01,
            "spectral radius of I must be 1, got {}",
            r
        );
    }

    /// Spectral radius of the zero matrix = 0.0.
    #[test]
    fn raio_espectral_zero_e_0() {
        let n = 3usize;
        let m = vec![0.0_f32; n * n];
        let r = spectral_radius(&m, n, 50);
        assert!(r < 1e-10, "spectral radius of zeros must be 0, got {}", r);
    }

    /// Spectral radius of scalar 2 = 2.0.
    #[test]
    fn raio_espectral_escalar_2() {
        // Matrix [[2.0]]: eigenvalue = 2.0
        let r = spectral_radius(&[2.0], 1, 10);
        assert!(
            (r - 2.0).abs() < 0.01,
            "spectral radius of [[2]] must be 2, got {}",
            r
        );
    }

    /// Effective rank of vector [1,0,0,...]: only 1 singular value → rank = 1.
    #[test]
    fn effective_rank_dominante() {
        let sv = [1.0_f32, 0.001, 0.001, 0.001];
        let r = effective_rank(&sv, 0.95);
        // σ₀² = 1.0, total ≈ 1.0 → rank = 1 covers ≥ 95%
        assert_eq!(r, 1, "effective rank with a dominant value must be 1");
    }

    /// Effective rank of a uniform vector: all contribute → rank = N.
    #[test]
    fn effective_rank_uniforme_e_n() {
        let sv = [1.0_f32, 1.0, 1.0, 1.0];
        let r = effective_rank(&sv, 0.95);
        // Accumulated energy: 25%, 50%, 75%, 100% → rank = 4
        assert_eq!(r, 4, "effective rank of uniform σ must be N");
    }

    /// Effective rank of an empty vector = 0.
    #[test]
    fn effective_rank_vazio() {
        assert_eq!(effective_rank(&[], 0.95), 0);
    }

    /// Eigenvalues of a diagonal matrix are the diagonal elements themselves.
    #[test]
    fn eigenvalues_symmetric_diagonal() {
        let n = 3usize;
        let mut m = vec![0.0f32; n * n];
        let diag = [4.0f32, 1.0, 9.0];
        for i in 0..n {
            m[i * n + i] = diag[i];
        }
        let mut eig = eigenvalues_symmetric(&m, n);
        eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut esperado = diag.to_vec();
        esperado.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in eig.iter().zip(esperado.iter()) {
            assert!((a - b).abs() < 1e-3, "eigenvalue {} != expected {}", a, b);
        }
    }

    /// `largest_singular_value` of an orthonormal matrix (4×4 identity) = 1.0.
    #[test]
    fn largest_singular_value_identidade_e_1() {
        let n = 4usize;
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        let sv = largest_singular_value(&w, n, n);
        assert!((sv - 1.0).abs() < 0.05, "expected ~1.0, got {sv}");
    }

    /// Rectangular matrix (2×4, rows < cols, the L1 192→96 case in practice):
    /// two orthogonal rows with different norms → largest singular value = largest norm.
    #[test]
    fn largest_singular_value_retangular() {
        let w = vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let sv = largest_singular_value(&w, 2, 4);
        assert!((sv - 3.0).abs() < 0.05, "expected ~3.0, got {sv}");
    }

    /// Scaling the whole matrix by `c` scales the largest singular value by the same `c`
    /// (linear) — it is exactly the property that the genome's spectral initialization
    /// (P0.5) uses to hit the target: `factor = target / measured`.
    #[test]
    fn largest_singular_value_escala_linear_com_a_matriz() {
        let base = vec![0.6_f32, -0.2, 0.1, 0.4, -0.5, 0.3, 0.2, -0.1];
        let sv_base = largest_singular_value(&base, 2, 4);
        let escalado: Vec<f32> = base.iter().map(|&v| v * 2.5).collect();
        let sv_escalado = largest_singular_value(&escalado, 2, 4);
        assert!(
            (sv_escalado - sv_base * 2.5).abs() < 1e-3,
            "expected {} (= {} × 2.5), got {}",
            sv_base * 2.5,
            sv_base,
            sv_escalado
        );
    }

    /// Wrong shape (`w.len() != rows*cols`) returns 0.0 instead of panicking.
    #[test]
    fn largest_singular_value_forma_errada_retorna_zero() {
        assert_eq!(largest_singular_value(&[1.0, 2.0, 3.0], 2, 2), 0.0);
        assert_eq!(largest_singular_value(&[], 0, 5), 0.0);
    }

    /// The sum of the eigenvalues is the trace — a cheap cross-check independent
    /// of the Jacobi converging in any specific order.
    #[test]
    fn eigenvalues_symmetric_soma_e_traco() {
        let n = 3usize;
        // Arbitrary symmetric matrix.
        let m = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let traco: f32 = (0..n).map(|i| m[i * n + i]).sum();
        let eig = eigenvalues_symmetric(&m, n);
        let soma: f32 = eig.iter().sum();
        assert!(
            (soma - traco).abs() < 1e-2,
            "eigenvalue sum {} != trace {}",
            soma,
            traco
        );
    }
}
