//! The project's single deterministic generator.
//!
//! # Why this file exists
//!
//! The same LCG was hand-written in 7 places (`engine.rs` ×4, `gpu_validate.rs` ×2,
//! `dna_genesis.rs` ×1) and **all with the same sign error**:
//!
//! ```text
//! ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
//! //  \_____ 31 bits _____/   \__ 2^31 __/
//! //  the quotient is already in [0,1) — subtracting 1 gives [-1,0), never positive
//! ```
//!
//! The `* 2.0` was missing. Consequence measured in `slow_probe`: **1024/1024 weights of
//! `dense1` and 48/48 of `dense2` were born negative**, which turns each layer into a
//! fixed-sign adder — the three outputs of the slow brain became the SAME sum
//! (correlation +0.9999, first principal component explaining 100% of the variance) and
//! never positive, leaving `throttle_cap` stuck at 1.0 permanently.
//!
//! A sign bug copied 7 times is the argument for having this in one place: now there is
//! a test that fails if the symmetry ever breaks again.

/// 64-bit LCG (Knuth/PCG constants), deterministic and with no external dependency.
/// Used where reproducibility matters more than statistical quality: weight
/// initialization, denoise noise, test fixtures.
#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Uniform in `[-1, 1)` — **zero mean**. This is the one weight initialization wants.
    #[inline]
    pub fn symmetric(&mut self) -> f32 {
        self.uniform() * 2.0 - 1.0
    }

    /// Normal(0, 1) via Box-Muller. Was copied 4× in `spawn.rs`, always identical.
    #[inline]
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// `true` with probability `p`.
    #[inline]
    pub fn bernoulli(&mut self, p: f32) -> bool {
        self.uniform() < p
    }

    /// Vector of symmetric weights with scale `sigma`. Shortcut for the most common case.
    pub fn weights(&mut self, n: usize, sigma: f32) -> Vec<f32> {
        (0..n).map(|_| self.symmetric() * sigma).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test the project did not have. It fails with the old idiom (`- 1.0` without
    /// the `* 2.0`), which is exactly how the bug survived 7 copies.
    #[test]
    fn symmetric_covers_both_signs() {
        let mut r = Lcg::new(0xDEAD_BEEF);
        let samples: Vec<f32> = (0..10_000).map(|_| r.symmetric()).collect();
        let positives = samples.iter().filter(|&&v| v > 0.0).count();
        assert!(
            (4_000..6_000).contains(&positives),
            "asymmetric init: {positives}/10000 positives — the sign bug is back"
        );
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean} far from zero");
    }

    #[test]
    fn uniform_stays_in_unit_interval() {
        let mut r = Lcg::new(7);
        for _ in 0..10_000 {
            let v = r.uniform();
            assert!((0.0..1.0).contains(&v), "uniform outside [0,1): {v}");
        }
    }

    #[test]
    fn symmetric_has_uniform_variance() {
        // Var(U[-1,1]) = 1/3. With the bug ([-1,0)) it would give 1/12 — which is why
        // the DCT test in dna_genesis.rs passed for the wrong reason.
        let mut r = Lcg::new(99);
        let a: Vec<f32> = (0..50_000).map(|_| r.symmetric()).collect();
        let m = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - m).powi(2)).sum::<f32>() / a.len() as f32;
        assert!(
            (var - 1.0 / 3.0).abs() < 0.02,
            "variance {var}, expected 1/3"
        );
    }

    #[test]
    fn gaussian_has_zero_mean_and_unit_stddev() {
        let mut r = Lcg::new(0xC0FFEE);
        let a: Vec<f32> = (0..50_000).map(|_| r.gaussian()).collect();
        let m = a.iter().sum::<f32>() / a.len() as f32;
        let sd = (a.iter().map(|v| (v - m).powi(2)).sum::<f32>() / a.len() as f32).sqrt();
        assert!(m.abs() < 0.03, "mean {m}");
        assert!((sd - 1.0).abs() < 0.05, "stddev {sd}");
    }

    #[test]
    fn is_deterministic_for_same_seed() {
        let a: Vec<f32> = {
            let mut r = Lcg::new(42);
            (0..100).map(|_| r.symmetric()).collect()
        };
        let b: Vec<f32> = {
            let mut r = Lcg::new(42);
            (0..100).map(|_| r.symmetric()).collect()
        };
        assert_eq!(a, b);
    }
}
