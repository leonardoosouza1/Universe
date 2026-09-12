//! The τ bank the CfC reads, under three parametrisations of the same 96 numbers.
//!
//! `Genome::tau` carries `N_CFC_CHANNELS` decay RATES in s⁻¹ (τ = 1/d), one per CfC
//! channel, and every one of them is a free gene. This module asks the question that
//! 96 free genes cannot answer on their own: **is the bank carrying structure, or is
//! it carrying 96 independent numbers?** It does so by rebuilding the bank, at upload
//! time, from a family with far fewer degrees of freedom:
//!
//! | mode | genes used | what the bank is |
//! |---|---|---|
//! | [`TauBankMode::Free`] | 96 | today's behaviour, bit for bit |
//! | [`TauBankMode::LogFixed`] | 1 | a fixed log-spaced ladder, scaled |
//! | [`TauBankMode::Spectral`] | 2 | offset + a 1-D Laplacian spectrum |
//!
//! Why at UPLOAD time and never in `Genome::tau` itself: the genome stays the genome.
//! Checkpoints written under one mode load under another, `--retomar` keeps working
//! across modes, and the mutation operator in `evolve()` is untouched. The non-free
//! modes are a LENS on the genes, not a rewrite of them — which also means the genes
//! the lens ignores keep mutating and keep being carried on disk, costing nothing but
//! bytes.
//!
//! A caveat that has to be stated with numbers, because it bounds what the two
//! structured modes can express. The scale genes are read out of `tau`, and `tau`
//! lives in rate space: it is born at [`crate::cfc::DECAY_DEFAULT`] = 0.5 and
//! `evolve()` clamps it to `[1/TAU_MAX_S, 1/TAU_MIN_S]` = [0.25, 40] s⁻¹. So a gene
//! read as an unconstrained real is in fact confined to [0.25, 40]:
//!
//! * [`TauBankMode::LogFixed`]'s scale is `exp(tau[0])` ∈ [1.28, e⁴⁰]. At birth
//!   (`tau[0]` = 0.5) it is 1.65, which already pushes the top of the ladder past the
//!   clamp — about the fastest 14 % of the channels pin at 40 s⁻¹ — and any upward
//!   drift of `tau[0]` collapses the whole bank onto the ceiling. The mode can make
//!   the bank uniformly faster; it cannot make it uniformly slower than 1.28× the
//!   ladder.
//! * [`TauBankMode::Spectral`]'s diffusion is `sigmoid(tau[1])` ∈ [0.562, 1.0], never
//!   the 0 that would flatten the bank. The flat case exists in the maths (and the
//!   tests exercise it) but not in the realizable gene range.
//!
//! Both follow from the two constraints the design is under — do not touch the
//! mutation rule, do not rewrite `Genome::tau` — and both are measurable rather than
//! hypothetical. If a battery shows `log-fixo` pinned at the ceiling from generation
//! 1, that is this paragraph, not a bug.

use std::sync::atomic::{AtomicU32, Ordering};

/// Fastest admitted decay rate, s⁻¹. Single source: [`crate::cfc::TAU_MIN_S`].
pub const D_HI: f32 = 1.0 / crate::cfc::TAU_MIN_S;
/// Slowest admitted decay rate, s⁻¹. Single source: [`crate::cfc::TAU_MAX_S`].
pub const D_LO: f32 = 1.0 / crate::cfc::TAU_MAX_S;

/// How the decay-rate bank uploaded to the GPU is built from the genome's `tau` genes.
///
/// Selected once per run by `gpu_evolve --banco-tau <mode>`; [`TauBankMode::Free`] is
/// the default and the control arm.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TauBankMode {
    /// `--banco-tau livre`: the bank IS `genome.tau`. Identity, no arithmetic.
    Free,
    /// `--banco-tau log-fixo`: fixed log-spaced ladder × one evolvable scale gene.
    LogFixed,
    /// `--banco-tau espectral`: offset + diffusion × a 1-D Laplacian spectrum.
    Spectral,
}

impl TauBankMode {
    /// Parses the CLI spelling. Returns `None` for anything else, so the caller can
    /// report the three valid values instead of silently running the control arm.
    pub fn from_flag(s: &str) -> Option<Self> {
        match s {
            "livre" => Some(Self::Free),
            "log-fixo" => Some(Self::LogFixed),
            "espectral" => Some(Self::Spectral),
            _ => None,
        }
    }

    /// The CLI spelling, for the run header. Round-trips with [`Self::from_flag`].
    pub fn as_flag(self) -> &'static str {
        match self {
            Self::Free => "livre",
            Self::LogFixed => "log-fixo",
            Self::Spectral => "espectral",
        }
    }

    /// One line for the run header: which genes the mode actually reads.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Free => "bank = genome.tau, 96 free genes (historical behaviour)",
            Self::LogFixed => "fixed log ladder [4 s .. 25 ms] scaled by exp(tau[0]) — 1 gene",
            Self::Spectral => {
                "lambda + D.mu_k, 1-D Laplacian spectrum over channel index — 2 genes"
            }
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Free => 0,
            Self::LogFixed => 1,
            Self::Spectral => 2,
        }
    }

    fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::LogFixed,
            2 => Self::Spectral,
            _ => Self::Free,
        }
    }
}

/// Process-wide mode, written once by `main()` and read by `Genome::pack_taus`.
///
/// It is a static for the same reason as `gpu::perception::BODY_CHANNELS_ON`: the
/// reader (`pack_taus`) is a library function called from three places, and threading
/// a config argument through all of them would change three signatures to carry a
/// value that is constant for the whole process.
static MODE: AtomicU32 = AtomicU32::new(0);

/// The mode in force. [`TauBankMode::Free`] until `main()` says otherwise.
pub fn mode() -> TauBankMode {
    TauBankMode::from_u32(MODE.load(Ordering::Relaxed))
}

/// Sets the mode. Called once, from CLI parsing, before any genome is uploaded.
pub fn set_mode(m: TauBankMode) {
    MODE.store(m.as_u32(), Ordering::Relaxed);
}

/// Builds the decay-rate bank (s⁻¹) the GPU will read, from the genome's `tau` genes.
///
/// Pure: same `mode` and `genes` in, same `Vec` out, no globals, no RNG. The output
/// has exactly `genes.len()` entries — `N_CFC_CHANNELS` = 96 at every real call site.
///
/// Every value except in [`TauBankMode::Free`] is clamped to `[D_LO, D_HI]`; `Free`
/// returns the genes untouched, because `evolve()` already clamps them there and a
/// clamp here would be a second source of truth for the same invariant.
pub fn banco_tau(mode: TauBankMode, genes: &[f32]) -> Vec<f32> {
    match mode {
        TauBankMode::Free => genes.to_vec(),
        TauBankMode::LogFixed => log_fixed_bank(genes),
        TauBankMode::Spectral => spectral_bank(genes),
    }
}

/// A fixed log-spaced ladder from `D_LO` to `D_HI`, multiplied by one scale gene.
///
///   `bank[k] = exp(ln(D_LO) + (ln(D_HI) - ln(D_LO)) · k/(n-1)) · exp(genes[0])`
///
/// The ladder itself is `log_uniform_decays`'s distribution with the randomness taken
/// out: the same coverage per decade, but deterministic and ordered. What evolution is
/// left to tune is a single number — where the whole ladder sits. `genes[0]` is read as
/// a LOG scale, so `genes[0] = 0` leaves the ladder exactly on the clamp range and a
/// step of ±0.69 halves or doubles every rate at once.
///
/// `genes[1..]` are ignored on purpose. They keep mutating in `evolve()` and keep being
/// saved; they simply do not reach the GPU in this mode.
fn log_fixed_bank(genes: &[f32]) -> Vec<f32> {
    let n = genes.len();
    if n == 0 {
        return Vec::new();
    }
    let scale = genes[0].exp();
    let lo = D_LO.ln();
    let hi = D_HI.ln();
    (0..n)
        .map(|k| {
            // n == 1 has no ladder to walk; put the single channel at the slow end so
            // the degenerate case is the slowest admitted τ, not a division by zero.
            let t = if n == 1 {
                0.0
            } else {
                k as f32 / (n - 1) as f32
            };
            ((lo + (hi - lo) * t).exp() * scale).clamp(D_LO, D_HI)
        })
        .collect()
}

/// Offset plus a diffusion coefficient times a 1-D Laplacian spectrum.
///
///   `mu[k] = ((k+1)/n)² · (D_HI - D_LO)`
///   `bank[k] = lambda + D · mu[k]`,  `lambda = softplus(genes[0]) + D_LO`,
///   `D = sigmoid(genes[1])`
///
/// `mu` is the eigenvalue spectrum of the Dirichlet Laplacian on a chain of `n` nodes,
/// whose eigenvalues grow like `k²` — the bank becomes "the k-th mode of a line of
/// channels", so a single diffusion coefficient sets how fast memory falls off along
/// the chain.
///
/// **The geometry is a CHOICE, not a measurement.** A CfC channel index is a position
/// in `perception::channels`, not a point on a line: ch40 is no more a neighbour of
/// ch41 than of ch90. Laying the 96 channels on a 1-D chain asserts a locality that the
/// channel registry does not have. It is worth testing precisely because it is a strong
/// assumption: if 2 genes under a wrong geometry still reach the free bank's fitness,
/// the free bank was not using its 96 degrees of freedom either.
///
/// `lambda` is the floor — every channel decays at least this fast — and `D` how much of
/// the admitted range the spectrum is allowed to climb. `D = 0` gives a flat bank at
/// `lambda`, i.e. the historical single-τ brain at an evolvable τ.
///
/// `genes[2..]` are ignored, same contract as `log_fixed_bank`.
fn spectral_bank(genes: &[f32]) -> Vec<f32> {
    let n = genes.len();
    if n == 0 {
        return Vec::new();
    }
    let lambda = (D_LO + softplus(genes[0])).clamp(D_LO, D_HI);
    // A one-channel bank has no spectrum to ride; it is the flat case by construction.
    let diffusion = if n >= 2 { sigmoid(genes[1]) } else { 0.0 };
    let span = D_HI - D_LO;
    (0..n)
        .map(|k| {
            let x = (k + 1) as f32 / n as f32;
            (lambda + diffusion * x * x * span).clamp(D_LO, D_HI)
        })
        .collect()
}

/// `ln(1 + eˣ)`, in the branch-stable form. The naive `(1+x.exp()).ln()` overflows to
/// `inf` around x = 89 in f32 and returns `inf` where the answer is 89.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `1/(1+e⁻ˣ)`, saturating cleanly at both ends.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The control arm has to be an identity, not "an identity up to rounding" — the
    /// whole value of the `livre` mode is that a battery run under it is comparable to
    /// every battery run before this module existed.
    #[test]
    fn free_mode_returns_the_genes_untouched() {
        for genes in [
            vec![],
            vec![0.5f32],
            vec![0.25, 40.0, 0.5, 1.0, 7.3, -1.0, f32::MAX],
            crate::cfc::log_uniform_decays(12345, 96),
        ] {
            let out = banco_tau(TauBankMode::Free, &genes);
            assert_eq!(out, genes, "livre must be bit-for-bit the input");
        }
    }

    #[test]
    fn log_fixed_spans_the_clamp_range_at_scale_zero() {
        let genes = vec![0.0f32; 96];
        let bank = banco_tau(TauBankMode::LogFixed, &genes);
        assert_eq!(bank.len(), 96);
        assert!(
            (bank[0] - D_LO).abs() < 1e-5,
            "ladder must start at 1/TAU_MAX_S = {D_LO} (got {})",
            bank[0]
        );
        assert!(
            (bank[95] - D_HI).abs() < 1e-3,
            "ladder must end at 1/TAU_MIN_S = {D_HI} (got {})",
            bank[95]
        );
    }

    #[test]
    fn log_fixed_is_monotone_in_k() {
        let genes = vec![0.0f32; 96];
        let bank = banco_tau(TauBankMode::LogFixed, &genes);
        for k in 1..bank.len() {
            assert!(
                bank[k] > bank[k - 1],
                "ladder must be strictly increasing at k={k}: {} !> {}",
                bank[k],
                bank[k - 1]
            );
        }
    }

    /// One gene, one effect: it multiplies every rate by the same factor. Checked below
    /// the clamp so the ceiling does not hide the property being tested.
    #[test]
    fn log_fixed_scale_gene_shifts_every_rate_by_one_factor() {
        let mut base = vec![0.0f32; 96];
        // -2.0 pulls the whole ladder down by e² = 7.39×, leaving the fast end at
        // 40/7.39 = 5.4 s⁻¹ — clear of D_HI, so nothing clamps and the ratio is exact.
        base[0] = -2.0;
        let mut lifted = base.clone();
        lifted[0] = -2.0 + 0.5;
        let a = banco_tau(TauBankMode::LogFixed, &base);
        let b = banco_tau(TauBankMode::LogFixed, &lifted);
        let expected = 0.5f32.exp();
        for k in 0..a.len() {
            // Skip anything that hit the slow floor; there the factor is by design lost.
            if a[k] <= D_LO * 1.001 {
                continue;
            }
            let ratio = b[k] / a[k];
            assert!(
                (ratio - expected).abs() < 1e-3,
                "channel {k} scaled by {ratio}, every channel must scale by {expected}"
            );
        }
    }

    #[test]
    fn log_fixed_ignores_every_gene_but_the_first() {
        let mut a = vec![0.0f32; 96];
        let mut b = a.clone();
        for (k, g) in b.iter_mut().enumerate().skip(1) {
            *g = k as f32;
        }
        a[0] = 0.3;
        b[0] = 0.3;
        assert_eq!(
            banco_tau(TauBankMode::LogFixed, &a),
            banco_tau(TauBankMode::LogFixed, &b),
            "genes 1.. must not reach the bank"
        );
    }

    #[test]
    fn spectral_is_monotone_in_k_and_clamped() {
        let mut genes = vec![0.0f32; 96];
        genes[0] = 0.5; // lambda
        genes[1] = 2.0; // D = sigmoid(2) = 0.881
        let bank = banco_tau(TauBankMode::Spectral, &genes);
        assert_eq!(bank.len(), 96);
        for k in 1..bank.len() {
            assert!(
                bank[k] >= bank[k - 1],
                "spectrum must not decrease at k={k}: {} < {}",
                bank[k],
                bank[k - 1]
            );
        }
        for (k, &d) in bank.iter().enumerate() {
            assert!(
                (D_LO..=D_HI).contains(&d),
                "channel {k} at {d} s^-1 is outside [{D_LO}, {D_HI}]"
            );
        }
    }

    /// `D = 0` is the historical brain: one τ on every channel. Reached here with a gene
    /// far below the realizable range (`evolve()` clamps `tau` to [0.25, 40], so the
    /// smallest realizable D is sigmoid(0.25) = 0.562) — the maths has the flat case even
    /// though the gene range does not, and this pins it.
    #[test]
    fn spectral_with_zero_diffusion_is_flat_at_lambda() {
        let mut genes = vec![0.0f32; 96];
        genes[0] = 0.5;
        genes[1] = -40.0; // sigmoid(-40) = 4.2e-18
        let bank = banco_tau(TauBankMode::Spectral, &genes);
        let lambda = D_LO + softplus(0.5);
        for (k, &d) in bank.iter().enumerate() {
            assert!(
                (d - lambda).abs() < 1e-6,
                "channel {k} at {d}, flat bank must sit at lambda = {lambda}"
            );
        }
    }

    #[test]
    fn spectral_lambda_is_the_floor_and_stays_in_range() {
        // Both extremes of the realizable gene range, plus one far outside it.
        for &g0 in &[0.25f32, 40.0, -50.0] {
            let mut genes = vec![0.0f32; 96];
            genes[0] = g0;
            genes[1] = 3.0;
            let bank = banco_tau(TauBankMode::Spectral, &genes);
            let lambda = (D_LO + softplus(g0)).clamp(D_LO, D_HI);
            assert!(
                bank[0] >= lambda - 1e-4,
                "lambda = {lambda} must be a floor, channel 0 came out {}",
                bank[0]
            );
            assert!(bank.iter().all(|&d| (D_LO..=D_HI).contains(&d)));
        }
    }

    #[test]
    fn spectral_ignores_every_gene_but_the_first_two() {
        let mut a = vec![0.0f32; 96];
        let mut b = a.clone();
        for (k, g) in b.iter_mut().enumerate().skip(2) {
            *g = k as f32;
        }
        a[0] = 0.5;
        a[1] = 1.0;
        b[0] = 0.5;
        b[1] = 1.0;
        assert_eq!(
            banco_tau(TauBankMode::Spectral, &a),
            banco_tau(TauBankMode::Spectral, &b),
            "genes 2.. must not reach the bank"
        );
    }

    #[test]
    fn flag_spelling_round_trips() {
        for m in [
            TauBankMode::Free,
            TauBankMode::LogFixed,
            TauBankMode::Spectral,
        ] {
            assert_eq!(TauBankMode::from_flag(m.as_flag()), Some(m));
        }
        assert_eq!(TauBankMode::from_flag("laplaciano"), None);
    }

    /// The default has to be the control arm, or a run that forgets the flag silently
    /// measures a different brain.
    #[test]
    fn default_mode_is_free() {
        assert_eq!(TauBankMode::from_u32(0), TauBankMode::Free);
        for m in [
            TauBankMode::Free,
            TauBankMode::LogFixed,
            TauBankMode::Spectral,
        ] {
            assert_eq!(TauBankMode::from_u32(m.as_u32()), m);
        }
    }

    #[test]
    fn softplus_does_not_overflow_at_the_top_of_the_gene_range() {
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        assert!(softplus(100.0).is_finite(), "naive form returns inf here");
        assert!((softplus(100.0) - 100.0).abs() < 1e-3);
        assert!(softplus(-100.0) >= 0.0);
    }
}
