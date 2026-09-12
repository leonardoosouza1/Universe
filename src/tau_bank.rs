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
//! **Every gene is read in RATE SPACE, never through an exponential.** This is the one
//! thing that makes the three arms comparable, and getting it wrong once cost this
//! module a rewrite. A `tau` gene is a decay rate in s⁻¹: born at
//! [`crate::cfc::DECAY_DEFAULT`] = 0.5, mutated MULTIPLICATIVELY in log by `evolve()`
//! (`d ← d·e^{σg}`, σ 0.20) and clamped by it to `[1/TAU_MAX_S, 1/TAU_MIN_S]` =
//! [0.25, 40]. So the maps are ratios and affine maps of that range, and each one is
//! neutral at birth:
//!
//! * [`TauBankMode::LogFixed`]'s scale is `tau[0] / DECAY_DEFAULT` — **1.0 at birth**,
//!   so a newborn bank is the untouched ladder spanning [0.25, 40] end to end, and the
//!   reachable range is [0.5, 80]. Because the gene is mutated by a ratio, one mutation
//!   step multiplies every rate in the bank by that same ratio.
//! * [`TauBankMode::Spectral`]'s `lambda` is `tau[0]` itself (no map at all — it is
//!   already a rate, and at birth it puts the bank's floor exactly at the historical
//!   τ = 2 s), and its diffusion is
//!   `clamp((tau[1] - D_LO) / (2·(DECAY_DEFAULT - D_LO)), 0, 1)` — **0 at the gene
//!   floor** (so the flat, single-τ bank is reachable, not just a limit), 0.5 at birth,
//!   1 from `tau[1]` = 0.75 s⁻¹ up.

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
    /// `--banco-tau log-fixo`: fixed log-spaced ladder × `tau[0]/DECAY_DEFAULT`.
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
            Self::LogFixed => {
                "fixed log ladder [4 s .. 25 ms] scaled by tau[0]/0.5 (1.0 at birth) — 1 gene"
            }
            Self::Spectral => {
                "lambda=tau[0] + D.mu_k, 1-D Laplacian spectrum over channel index — 2 genes"
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
///   `bank[k] = clamp(exp(ln(D_LO) + (ln(D_HI) - ln(D_LO)) · k/(n-1)) · genes[0]/DECAY_DEFAULT)`
///
/// The ladder itself is `log_uniform_decays`'s distribution with the randomness taken
/// out: the same coverage per decade, but deterministic and ordered. What evolution is
/// left to tune is a single number — where the whole ladder sits.
///
/// The scale is a RATIO, `genes[0] / DECAY_DEFAULT`, because `genes[0]` is a decay RATE
/// like every other entry of `tau`: born at [`crate::cfc::DECAY_DEFAULT`] = 0.5 s⁻¹ and
/// mutated multiplicatively in log by `evolve()`. So a newborn genome has scale exactly
/// 1.0 and the bank IS the untouched ladder, spanning [0.25, 40] s⁻¹ end to end; the
/// gene's own clamp `[0.25, 40]` puts the reachable scale in [0.5, 80], i.e. the whole
/// ladder can be halved or multiplied by 80, and a mutation step that multiplies the
/// gene by 1.15 multiplies every rate in the bank by 1.15.
///
/// `genes[1..]` are ignored on purpose. They keep mutating in `evolve()` and keep being
/// saved; they simply do not reach the GPU in this mode.
fn log_fixed_bank(genes: &[f32]) -> Vec<f32> {
    let n = genes.len();
    if n == 0 {
        return Vec::new();
    }
    let scale = genes[0] / crate::cfc::DECAY_DEFAULT;
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
///   `bank[k] = clamp(lambda + D · mu[k])`,  `lambda = clamp(genes[0])`,
///   `D = clamp((genes[1] - D_LO) / (2·(DECAY_DEFAULT - D_LO)), 0, 1)`
///
/// Both genes are decay RATES (born at `DECAY_DEFAULT` = 0.5 s⁻¹, clamped by `evolve()`
/// to [0.25, 40]), so neither is passed through an exponential. `lambda` needs no map at
/// all — it already IS a rate, and a newborn genome puts the bank's floor exactly at the
/// historical τ = 2 s. `D` is the affine map of the gene's own range onto [0, 1] that
/// hits **0 at the gene floor** (`genes[1]` = `D_LO` = 0.25, the flat bank) and **0.5 at
/// birth** (`genes[1]` = 0.5); it saturates at 1 from `genes[1]` = 0.75 s⁻¹ up. The
/// denominator is `2·(DECAY_DEFAULT - D_LO)` = 0.5 and not `2·DECAY_DEFAULT - D_LO`
/// = 0.75, because only the first gives both fixed points — the second would put birth
/// at 1/3, and "birth = half diffusion" is the property this map exists to have.
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
    let lambda = genes[0].clamp(D_LO, D_HI);
    // A one-channel bank has no spectrum to ride; it is the flat case by construction.
    let diffusion = if n >= 2 { diffusion_from(genes[1]) } else { 0.0 };
    let span = D_HI - D_LO;
    (0..n)
        .map(|k| {
            let x = (k + 1) as f32 / n as f32;
            (lambda + diffusion * x * x * span).clamp(D_LO, D_HI)
        })
        .collect()
}

/// Maps a decay-rate gene onto the diffusion coefficient `D ∈ [0, 1]`.
///
/// `0` at the gene floor `D_LO` = 0.25 s⁻¹ (the flat bank is reachable, not merely a
/// limit), `0.5` at birth (`DECAY_DEFAULT` = 0.5 s⁻¹), `1` from 0.75 s⁻¹ up. Exposed as
/// its own function only so the tests can state the fixed points without restating the
/// arithmetic.
pub fn diffusion_from(gene: f32) -> f32 {
    ((gene - D_LO) / (2.0 * (crate::cfc::DECAY_DEFAULT - D_LO))).clamp(0.0, 1.0)
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

    /// Birth is the fixed point that makes the arm interpretable: a newborn genome
    /// (every gene at `DECAY_DEFAULT`) has scale exactly 1.0, so the bank is the raw
    /// ladder and spans the whole admitted range with nothing clamped away.
    #[test]
    fn log_fixed_at_birth_is_the_untouched_ladder() {
        let genes = vec![crate::cfc::DECAY_DEFAULT; 96];
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
        let genes = vec![crate::cfc::DECAY_DEFAULT; 96];
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

    /// One gene, one effect: multiplying it by 1.5 multiplies every rate by 1.5 — which
    /// is exactly what `evolve()`'s multiplicative mutation does to it. Entries sitting
    /// on either clamp are skipped, because there the factor is by design lost.
    #[test]
    fn log_fixed_scale_gene_shifts_every_rate_by_one_factor() {
        let mut base = vec![crate::cfc::DECAY_DEFAULT; 96];
        base[0] = crate::cfc::DECAY_DEFAULT; // scale 1.0
        let mut lifted = base.clone();
        lifted[0] = crate::cfc::DECAY_DEFAULT * 1.5; // scale 1.5
        let a = banco_tau(TauBankMode::LogFixed, &base);
        let b = banco_tau(TauBankMode::LogFixed, &lifted);
        let mut checked = 0;
        for k in 0..a.len() {
            if a[k] <= D_LO * 1.001 || b[k] >= D_HI * 0.999 {
                continue;
            }
            let ratio = b[k] / a[k];
            assert!(
                (ratio - 1.5).abs() < 1e-3,
                "channel {k} scaled by {ratio}, every channel must scale by 1.5"
            );
            checked += 1;
        }
        assert!(
            checked > 80,
            "only {checked} channels were clear of both clamps — the test stopped testing"
        );
    }

    #[test]
    fn log_fixed_ignores_every_gene_but_the_first() {
        let mut a = vec![crate::cfc::DECAY_DEFAULT; 96];
        let mut b = a.clone();
        for (k, g) in b.iter_mut().enumerate().skip(1) {
            *g = 0.25 + k as f32;
        }
        a[0] = 0.6;
        b[0] = 0.6;
        assert_eq!(
            banco_tau(TauBankMode::LogFixed, &a),
            banco_tau(TauBankMode::LogFixed, &b),
            "genes 1.. must not reach the bank"
        );
    }

    /// The three fixed points of the diffusion map, stated as numbers so a change to
    /// `DECAY_DEFAULT` or to the map cannot slide past review.
    #[test]
    fn diffusion_map_hits_zero_at_the_gene_floor_and_half_at_birth() {
        assert_eq!(diffusion_from(D_LO), 0.0, "gene floor must give D = 0");
        assert!(
            (diffusion_from(crate::cfc::DECAY_DEFAULT) - 0.5).abs() < 1e-6,
            "birth must give D = 0.5, got {}",
            diffusion_from(crate::cfc::DECAY_DEFAULT)
        );
        assert_eq!(diffusion_from(0.75), 1.0, "D saturates at 0.75 s^-1");
        assert_eq!(diffusion_from(D_HI), 1.0, "gene ceiling stays inside [0, 1]");
        assert_eq!(diffusion_from(-10.0), 0.0, "below the floor still clamps");
    }

    /// A newborn genome puts the bank's floor at the historical τ = 2 s and rides half
    /// the spectrum — neutral in the one sense that matters (`lambda` = `DECAY_DEFAULT`),
    /// and not saturated in either direction.
    #[test]
    fn spectral_at_birth_has_the_historical_floor_and_half_diffusion() {
        let genes = vec![crate::cfc::DECAY_DEFAULT; 96];
        let bank = banco_tau(TauBankMode::Spectral, &genes);
        let span = D_HI - D_LO;
        assert!(
            (bank[0] - (crate::cfc::DECAY_DEFAULT + 0.5 * span / (96.0 * 96.0))).abs() < 1e-4,
            "channel 0 came out {}",
            bank[0]
        );
        assert!(
            (bank[95] - (crate::cfc::DECAY_DEFAULT + 0.5 * span)).abs() < 1e-3,
            "channel 95 came out {} (expected lambda + 0.5.span)",
            bank[95]
        );
    }

    #[test]
    fn spectral_is_monotone_in_k_and_clamped() {
        let mut genes = vec![crate::cfc::DECAY_DEFAULT; 96];
        genes[0] = 0.5; // lambda = 0.5 s^-1, the historical tau = 2 s
        genes[1] = 0.5; // D = 0.5 at birth
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

    /// `D = 0` is the historical brain: one τ on every channel. It is REACHABLE — the
    /// gene floor `evolve()` clamps to, 0.25 s⁻¹, maps to exactly 0 — so `espectral` can
    /// evolve its way back to the single-τ bank instead of only approaching it.
    #[test]
    fn spectral_with_zero_diffusion_is_flat_at_lambda() {
        let mut genes = vec![crate::cfc::DECAY_DEFAULT; 96];
        genes[0] = 0.5;
        genes[1] = D_LO; // the gene's own floor, not an unreachable value
        let bank = banco_tau(TauBankMode::Spectral, &genes);
        for (k, &d) in bank.iter().enumerate() {
            assert_eq!(
                d, 0.5,
                "channel {k} at {d}, flat bank must sit at lambda = 0.5"
            );
        }
    }

    #[test]
    fn spectral_lambda_is_the_floor_and_stays_in_range() {
        // Both extremes of the realizable gene range, plus one far outside it.
        for &g0 in &[0.25f32, 40.0, -50.0] {
            let mut genes = vec![crate::cfc::DECAY_DEFAULT; 96];
            genes[0] = g0;
            genes[1] = 0.75; // D = 1, the spectrum climbs as far as it can
            let bank = banco_tau(TauBankMode::Spectral, &genes);
            let lambda = g0.clamp(D_LO, D_HI);
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
        let mut a = vec![crate::cfc::DECAY_DEFAULT; 96];
        let mut b = a.clone();
        for (k, g) in b.iter_mut().enumerate().skip(2) {
            *g = 0.25 + k as f32;
        }
        a[0] = 0.5;
        a[1] = 0.6;
        b[0] = 0.5;
        b[1] = 0.6;
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
}
