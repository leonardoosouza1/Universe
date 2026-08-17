/// Measured criterion to decide whether a brain deserves to grow.
///
/// Two conditions (both must hold to justify growth):
///
/// (A) SATURATION — Participation Ratio (PR) of the layer's activations.
///     PR = (Σλ_i)² / Σλ_i²  where λ_i are eigenvalues of the covariance.
///     PR ≈ width ⇒ all dimensions in use (saturated, growing may be worth it).
///     PR << width ⇒ idle dimensions (growing is just wasted space).
///
/// (B) REAL GAIN — Prospecting Law ("Lei do Garimpo") of the IARA ecosystem:
///     Picking the best of M alternatives of pure noise, the expected maximum is
///     σ·√(2·ln M). A fitness gain is only real if it exceeds this threshold.

/// Growth verdict, with the reason broken out.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Growing is justified: capacity saturated AND gain exceeds search noise.
    Grow,
    /// There are idle dimensions — growing wastes memory/compute.
    WastedSpace,
    /// Capacity saturated but the gain does not exceed the Prospecting threshold.
    GainNotReal,
    /// Insufficient samples to estimate the covariance with confidence
    /// (requires at least `width + 1` samples).
    InsufficientData,
}

// ---------------------------------------------------------------------------
// Cyclic Jacobi for eigenvalues of a symmetric NxN matrix (no dependencies).
// ---------------------------------------------------------------------------

/// Returns the N eigenvalues of a symmetric NxN matrix (row-major).
/// Uses cyclic Jacobi: rotates every pair (i,j) i<j in sequence,
/// repeating until the off-diagonal norm is < tol or max_iter is reached.
/// For small matrices (<= 32×32) it converges in dozens of sweeps.
fn jacobi_eigenvalues(mat: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(mat.len(), n * n);
    let mut a: Vec<f32> = mat.to_vec(); // mutable copy
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

        // One sweep: rotates every pair (p, q) with p < q
        for p in 0..n {
            for q in (p + 1)..n {
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                // Jacobi rotation angle
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                // Update diagonal
                let dpp = t * apq;
                a[p * n + p] -= dpp;
                a[q * n + q] += dpp;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update rows/columns r != p, r != q
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

    // Diagonal = eigenvalues
    (0..n).map(|i| a[i * n + i]).collect()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes the Participation Ratio of a layer's activations.
///
/// `activations`: rows = samples over time, columns = layer units.
/// Centers each column, computes the empirical covariance, extracts eigenvalues,
/// returns PR = (Σλ_i)² / Σλ_i².
///
/// Returns `None` if there are insufficient samples (T <= width) or width == 0.
pub fn participation_ratio(activations: &[Vec<f32>]) -> Option<f32> {
    let t = activations.len();
    if t == 0 {
        return None;
    }
    let width = activations[0].len();
    if width == 0 || t <= width {
        // Needs more samples than dimensions for a well-defined covariance
        return None;
    }

    // Per-column means
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

    // Empirical covariance matrix (width × width), divided by T (not T-1)
    // so PR is exactly `width` when all columns are iid with equal variance.
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

    // Positive eigenvalues only (numeric noise can produce slightly negative ones)
    let lambdas: Vec<f32> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();

    let sum1: f32 = lambdas.iter().sum();
    let sum2: f32 = lambdas.iter().map(|v| v * v).sum();

    if sum2 < 1e-30 {
        // All activations constant — rank zero, PR = 1 by convention
        return Some(1.0);
    }

    Some(sum1 * sum1 / sum2)
}

/// Saturation = PR / layer width, in [0, 1].
/// 1.0 ⇒ all dimensions actively used (saturated).
/// 0.0 ⇒ effective rank close to 1 (one dimension dominates everything).
pub fn saturation(activations: &[Vec<f32>], width: usize) -> Option<f32> {
    if width == 0 {
        return None;
    }
    participation_ratio(activations).map(|pr| (pr / width as f32).min(1.0))
}

/// Prospecting threshold: expected gain of the best of M samples of pure N(0, σ²) noise.
/// Formula: σ · √(2 · ln M).
/// If σ == 0 or M <= 1, returns 0.0 (no gain is "real" to check).
pub fn search_noise_threshold(sigma: f32, m_alternatives: usize) -> f32 {
    if sigma <= 0.0 || m_alternatives <= 1 {
        return 0.0;
    }
    sigma * (2.0 * (m_alternatives as f32).ln()).sqrt()
}

/// Combines condition (A) saturation and condition (B) Prospecting threshold into one verdict.
///
/// Parameters:
/// - `activations`: activation history of the layer under evaluation.
/// - `width`: nominal layer width (number of units).
/// - `saturation_threshold`: PR/width above which we call it saturated (e.g. 0.85).
/// - `fitness_gain`: fitness improvement observed after the M evaluations.
/// - `fitness_sigma`: standard deviation of fitness in the population.
/// - `m_alternatives`: total number of variants evaluated (pop × generations).
pub fn growth_verdict(
    activations: &[Vec<f32>],
    width: usize,
    saturation_threshold: f32,
    fitness_gain: f32,
    fitness_sigma: f32,
    m_alternatives: usize,
) -> Verdict {
    // Insufficient samples to estimate PR
    let sat = match saturation(activations, width) {
        Some(s) => s,
        None => return Verdict::InsufficientData,
    };

    // (A) capacity saturated?
    if sat < saturation_threshold {
        return Verdict::WastedSpace;
    }

    // (B) gain exceeds the Prospecting threshold?
    let threshold = search_noise_threshold(fitness_sigma, m_alternatives);
    if fitness_gain <= threshold {
        return Verdict::GainNotReal;
    }

    Verdict::Grow
}

// ---------------------------------------------------------------------------
// Unit tests with synthetic data of known properties
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds T samples where every column is a multiple of column 0.
    /// Effective rank = 1 ⇒ PR must be ≈ 1.
    #[test]
    fn pr_rank1_matrix() {
        let t = 100usize;
        let width = 4usize;
        // Column j = (j+1) * column 0; column 0 varies from -1 to 1 linearly.
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let base = (i as f32 / (t - 1) as f32) * 2.0 - 1.0; // [-1, 1]
                (0..width).map(|j| base * (j + 1) as f32).collect()
            })
            .collect();
        let pr = participation_ratio(&activations).expect("must have PR");
        // Rank 1 ⇒ PR = 1
        assert!(
            (pr - 1.0).abs() < 0.05,
            "PR of a rank-1 matrix must be ≈ 1, got {}",
            pr
        );
    }

    /// k independent columns with equal variance ⇒ PR = k.
    #[test]
    fn pr_independent_equal_variance() {
        let t = 500usize;
        let width = 6usize;
        // Deterministic orthogonal signal: column j = sin(2π·j·i/T)
        // For large T and different frequencies they are approximately orthogonal.
        // We use prime frequencies to guarantee empirical orthogonality.
        let freqs: [f32; 6] = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0];
        let activations: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                freqs
                    .iter()
                    .map(|&f| (2.0 * std::f32::consts::PI * f * i as f32 / t as f32).sin())
                    .collect()
            })
            .collect();
        let pr = participation_ratio(&activations).expect("must have PR");
        // Orthogonal sines of equal amplitude ⇒ PR ≈ width = 6
        assert!(
            (pr - width as f32).abs() < 0.5,
            "PR of orthogonal equal-variance signals must be ≈ {}, got {}",
            width,
            pr
        );
    }

    /// Decaying spectrum (eigenvalues λ_i ∝ 1/i) ⇒ 1 < PR < k.
    #[test]
    fn pr_decaying_spectrum_between_1_and_k() {
        let t = 300usize;
        let width = 5usize;
        // Builds activations whose columns have variances 1, 1/2, 1/3, 1/4, 1/5
        // and are orthogonal (distinct frequencies).
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
        let pr = participation_ratio(&activations).expect("must have PR");
        assert!(
            pr > 1.0 && pr < width as f32,
            "PR with a decaying spectrum must lie between 1 and {}, got {}",
            width,
            pr
        );
    }

    /// garimpo_threshold grows strictly with M.
    #[test]
    fn search_noise_grows_with_m() {
        let sigma = 1.0;
        let t10 = search_noise_threshold(sigma, 10);
        let t100 = search_noise_threshold(sigma, 100);
        let t1000 = search_noise_threshold(sigma, 1000);
        assert!(t10 < t100, "threshold must grow: {} < {}", t10, t100);
        assert!(t100 < t1000, "threshold must grow: {} < {}", t100, t1000);
    }

    /// garimpo_threshold = 0 when σ = 0.
    #[test]
    fn search_noise_zero_when_sigma_zero() {
        assert_eq!(search_noise_threshold(0.0, 1000), 0.0);
    }

    /// garimpo_threshold = 0 when M <= 1.
    #[test]
    fn search_noise_zero_when_m_one() {
        assert_eq!(search_noise_threshold(1.5, 1), 0.0);
    }

    /// InsufficientData when there are few samples (T <= width).
    #[test]
    fn verdict_insufficient_data() {
        // width = 4, but only 3 samples
        let activations: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
        ];
        let v = growth_verdict(&activations, 4, 0.85, 1.0, 0.5, 100);
        assert_eq!(v, Verdict::InsufficientData);
    }

    /// WastedSpace when PR is far below the width.
    #[test]
    fn verdict_wasted_space() {
        // Rank 1 ⇒ saturation ≈ 1/4 = 0.25, below 0.85
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

    /// GainNotReal when capacity is saturated but gain is below the threshold.
    #[test]
    fn verdict_gain_not_real() {
        // Orthogonal signals ⇒ PR ≈ width (saturated)
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
        // sigma=1, M=100 ⇒ threshold = √(2·ln100) ≈ 3.03; gain=1.0 < 3.03
        let v = growth_verdict(&activations, width, 0.85, 1.0, 1.0, 100);
        assert_eq!(v, Verdict::GainNotReal);
    }

    /// Grow when saturated AND gain above the threshold.
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
        // sigma=1, M=100 ⇒ threshold ≈ 3.03; gain=10.0 > 3.03
        let v = growth_verdict(&activations, width, 0.85, 10.0, 1.0, 100);
        assert_eq!(v, Verdict::Grow);
    }

    /// Known numeric value of the Prospecting threshold.
    #[test]
    fn search_noise_threshold_known_value() {
        // σ=1, M=e ⇒ √(2·ln e) = √2 ≈ 1.4142
        let threshold = search_noise_threshold(1.0, std::f32::consts::E as usize);
        // E as usize = 2, ln(2) ≈ 0.693, √(2·0.693) ≈ 1.177
        let expected = (2.0 * (2.0_f32).ln()).sqrt();
        assert!(
            (threshold - expected).abs() < 1e-4,
            "expected {}, got {}",
            expected,
            threshold
        );
    }

    /// saturation() returns None when width=0.
    #[test]
    fn saturation_none_on_zero_width() {
        let activations: Vec<Vec<f32>> = vec![vec![]];
        assert!(saturation(&activations, 0).is_none());
    }
}
