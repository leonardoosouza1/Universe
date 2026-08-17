/// Descriptive statistics — mean, variance, percentiles, histogram.
///
/// All functions operate on f32 slices and do not allocate beyond the explicit outputs.
/// They are used by the `analysis/activation.rs` module (E4) to monitor neurons.

/// Arithmetic mean of a slice. Returns 0.0 if empty.
pub fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Sample variance (divisor N-1). Returns 0.0 if len < 2.
///
/// N-1 (Bessel) reduces bias for small populations. PR computations use
/// N (see brain_capacity.rs), but here the context is sample statistics.
pub fn variance(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / (xs.len() - 1) as f32
}

/// Sample standard deviation (square root of the N-1 variance).
pub fn std_dev(xs: &[f32]) -> f32 {
    variance(xs).sqrt()
}

/// POPULATION standard deviation (divisor N, no Bessel correction), via
/// sum-of-squares: `sqrt(E[x²] - E[x]²)`.
///
/// Mirrors `brain.wgsl` STAGE 2 (`params.norm_l2_by_std`, P1.5,
/// `docs/PLANO_EVOLUCAO.md`) — the shader computes `sum`/`sum_sq` in a single loop
/// (without allocating a second vector of deviations) because it runs in WGSL without a heap;
/// this function replicates the SAME algorithm, not `std_dev`/N-1, so that the test
/// in `math::activations` compares the same formula that is in the shader, not an
/// approximation. `max(0.0, ...)` because the sum-of-squares variance can
/// yield a negative epsilon from floating-point cancellation when the
/// values are nearly equal — same guard the shader has.
///
/// Returns 0.0 if empty (not 0.0/0.0 = NaN).
pub fn population_std_dev(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f32;
    let sum: f32 = xs.iter().sum();
    let sum_sq: f32 = xs.iter().map(|&x| x * x).sum();
    let m = sum / n;
    let variance = (sum_sq / n - m * m).max(0.0);
    variance.sqrt()
}

/// Percentile p ∈ [0.0, 100.0] using linear interpolation (R7 / Excel method).
///
/// Sorts the slice in-place — pass a copy if you need to preserve the order.
/// Returns 0.0 if empty.
pub fn percentile(xs: &mut [f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len() as f32;
    // Real index via the R7 method: idx = p/100 * (n-1)
    let idx = (p / 100.0) * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(xs.len() - 1);
    let frac = idx - lo as f32;
    xs[lo] + frac * (xs[hi] - xs[lo])
}

/// Histogram of `n_bins` uniform bins covering [min, max] of the data.
///
/// Returns a Vec<u32> of length `n_bins`.
/// If all values are equal, they all fall into the central bin (floor).
/// If `n_bins == 0` or the slice is empty, returns an empty Vec.
pub fn histogram(xs: &[f32], n_bins: usize) -> Vec<u32> {
    if n_bins == 0 || xs.is_empty() {
        return vec![];
    }
    let min = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let mut bins = vec![0u32; n_bins];
    for &v in xs {
        let idx = if range < 1e-30 {
            // All values equal: goes to the middle bin
            n_bins / 2
        } else {
            let frac = (v - min) / range;
            // frac ∈ [0.0, 1.0]; maps to [0, n_bins-1]
            ((frac * n_bins as f32).floor() as usize).min(n_bins - 1)
        };
        bins[idx] += 1;
    }
    bins
}

/// Median of the slice (percentile 50.0).
///
/// Sorts in-place — pass a copy if needed.
pub fn median(xs: &mut [f32]) -> f32 {
    percentile(xs, 50.0)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_de_vetor_uniforme() {
        assert_eq!(mean(&[2.0; 5]), 2.0);
    }

    #[test]
    fn mean_de_vazio_e_zero() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn mean_de_sequencia_simetrica() {
        // [-2, -1, 0, 1, 2] → mean = 0
        let xs: Vec<f32> = (-2..=2).map(|i| i as f32).collect();
        assert!((mean(&xs) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn variance_de_constante_e_zero() {
        assert_eq!(variance(&[3.0; 10]), 0.0);
    }

    #[test]
    fn variance_conhecida() {
        // {0, 1, 2, 3, 4} → mean 2, sample variance = 10/4 = 2.5
        let xs = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let v = variance(&xs);
        assert!((v - 2.5).abs() < 1e-5, "variance = {}", v);
    }

    #[test]
    fn variance_vazio_e_zero() {
        assert_eq!(variance(&[]), 0.0);
    }

    #[test]
    fn percentile_min_e_max() {
        let mut xs = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&mut xs, 0.0) - 1.0).abs() < 1e-5);
        assert!((percentile(&mut xs, 100.0) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn percentile_50_de_impar() {
        let mut xs = [3.0_f32, 1.0, 2.0]; // sorted: [1, 2, 3], median = 2.0
        let med = percentile(&mut xs, 50.0);
        assert!((med - 2.0).abs() < 1e-5, "median = {}", med);
    }

    #[test]
    fn histogram_soma_total() {
        // All values must be in some bin
        let xs = [0.0_f32, 0.5, 1.0];
        let h = histogram(&xs, 5);
        assert_eq!(h.iter().sum::<u32>(), 3);
    }

    #[test]
    fn histogram_distribuicao_uniforme() {
        // 100 equally spaced values in [0, 1] → 10 bins with ~10 each
        let xs: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let h = histogram(&xs, 10);
        assert_eq!(h.len(), 10);
        assert_eq!(h.iter().sum::<u32>(), 100);
        // Each bin should have approximately 10 ±3 (tolerant to edges)
        for (i, &count) in h.iter().enumerate() {
            assert!(
                count >= 7 && count <= 13,
                "bin {} has {} (expected ~10)",
                i,
                count
            );
        }
    }

    #[test]
    fn histogram_valores_iguais_nao_trava() {
        let xs = [5.0_f32; 20];
        let h = histogram(&xs, 8);
        assert_eq!(h.iter().sum::<u32>(), 20, "all must be in some bin");
    }

    #[test]
    fn histogram_vazio_retorna_vazio() {
        let h = histogram(&[], 5);
        assert!(h.is_empty());
    }

    #[test]
    fn histogram_n_bins_zero_retorna_vazio() {
        let h = histogram(&[1.0, 2.0, 3.0], 0);
        assert!(h.is_empty());
    }

    #[test]
    fn std_dev_de_constante_e_zero() {
        assert_eq!(std_dev(&[7.0; 5]), 0.0);
    }

    #[test]
    fn population_std_dev_de_constante_e_zero() {
        assert_eq!(population_std_dev(&[7.0; 48]), 0.0);
    }

    #[test]
    fn population_std_dev_vazio_e_zero() {
        assert_eq!(population_std_dev(&[]), 0.0);
    }

    #[test]
    fn population_std_dev_conhecido() {
        // {0,1,2,3} → mean 1.5, POPULATION variance (÷N) = 1.25, deviation ≈ 1.1180
        let xs = [0.0_f32, 1.0, 2.0, 3.0];
        let d = population_std_dev(&xs);
        assert!(
            (d - 1.25f32.sqrt()).abs() < 1e-5,
            "population deviation = {}",
            d
        );
    }

    #[test]
    fn population_std_dev_e_menor_que_amostral_para_mesmo_conjunto() {
        // Divisor N < N-1 → population deviation <= sample (N>1).
        let xs = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        assert!(population_std_dev(&xs) < std_dev(&xs));
    }

    #[test]
    fn median_de_tres_elementos() {
        let mut xs = [5.0_f32, 1.0, 3.0];
        assert!((median(&mut xs) - 3.0).abs() < 1e-5);
    }
}
