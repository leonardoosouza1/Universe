use std::fs::File;
use std::io::{Read, Write};

// ============================================================================
// TIER 6.6 — WEIGHT SCARS (trauma immutability)
// ----------------------------------------------------------------------------
// When an organism dies in a CATASTROPHE (frontal crash with HP→0), a fraction
// of the NON-ZERO weights (proxy for "weights active at that moment") is marked
// as "scar tissue" — blocked from mutation for N generations.
//
// - `scar_mask[i]` = TTL u8: 0 = free, >0 = locked (decrements each gen).
// - Scars survive the generation transition via EliteDna.
// - Not part of .dna v3 (runtime-only for now). Documented for v4.
//
// Simplified version (v1): proxy snapshot — active ≈ non-zero. The full version
// (v2, future) would use the BitLinear's last_activation_mask after a real forward.
// ============================================================================
pub const SCAR_TTL_INITIAL: u8 = 5; // locks for 5 generations
pub const SCAR_FRACTION: f32 = 0.10; // 10% of the non-zero weights become scars
pub const SCAR_CATASTROPHE_LABEL: &str = "BATIDA";

// ============================================================================
// TIER 6.3 LITE — MORPHOLOGICAL CO-EVOLUTION (body genes)
// ----------------------------------------------------------------------------
// 4 body parameters co-evolve with the brain. For now they are only
// STORED (in the DNA and in the Racer) and mutate along with the elites; the
// physical wiring is left for a follow-up.
//   - mass      (kg)     [700, 1400]  default 1000
//   - wheelbase (m)      [1.5, 3.0]   default 2.5
//   - downforce (coef)   [0.5, 2.0]   default 1.0
//   - grip_mult (coef)   [0.8, 1.2]   default 1.0
// Mutation: 5% chance per gen per gene, Gaussian noise σ = 5% of the range,
// clamp at the bounds.
// ============================================================================
pub const MORPHO_MASS_MIN: f32 = 700.0;
pub const MORPHO_MASS_MAX: f32 = 1400.0;
pub const MORPHO_WHEELBASE_MIN: f32 = 1.5;
pub const MORPHO_WHEELBASE_MAX: f32 = 3.0;
pub const MORPHO_DOWNFORCE_MIN: f32 = 0.5;
pub const MORPHO_DOWNFORCE_MAX: f32 = 2.0;
pub const MORPHO_GRIP_MIN: f32 = 0.8;
pub const MORPHO_GRIP_MAX: f32 = 1.2;
pub const MORPHO_MUTATION_RATE: f32 = 0.05; // 5% per gene per gen
pub const MORPHO_SIGMA_FRAC: f32 = 0.05; // σ = 5% of the range
pub const MORPHO_BYTES: usize = 16; // 4 × f32 LE

#[derive(Clone, Copy, Debug)]
pub struct MorphoGenes {
    pub mass: f32,
    pub wheelbase: f32,
    pub downforce: f32,
    pub grip_mult: f32,
}

impl Default for MorphoGenes {
    fn default() -> Self {
        Self {
            mass: 1000.0,
            wheelbase: 2.5,
            downforce: 1.0,
            grip_mult: 1.0,
        }
    }
}

impl MorphoGenes {
    pub fn to_le_bytes(&self) -> [u8; MORPHO_BYTES] {
        let mut out = [0u8; MORPHO_BYTES];
        out[0..4].copy_from_slice(&self.mass.to_le_bytes());
        out[4..8].copy_from_slice(&self.wheelbase.to_le_bytes());
        out[8..12].copy_from_slice(&self.downforce.to_le_bytes());
        out[12..16].copy_from_slice(&self.grip_mult.to_le_bytes());
        out
    }

    pub fn from_le_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < MORPHO_BYTES {
            return None;
        }
        let mass = f32::from_le_bytes(buf[0..4].try_into().ok()?);
        let wheelbase = f32::from_le_bytes(buf[4..8].try_into().ok()?);
        let downforce = f32::from_le_bytes(buf[8..12].try_into().ok()?);
        let grip_mult = f32::from_le_bytes(buf[12..16].try_into().ok()?);
        let out = Self {
            mass,
            wheelbase,
            downforce,
            grip_mult,
        };
        Some(out.sanitized())
    }

    /// Clamp to the ranges (+ protection against NaN/Inf coming from a corrupted file).
    pub fn sanitized(mut self) -> Self {
        let default = Self::default();
        if !self.mass.is_finite() {
            self.mass = default.mass;
        }
        if !self.wheelbase.is_finite() {
            self.wheelbase = default.wheelbase;
        }
        if !self.downforce.is_finite() {
            self.downforce = default.downforce;
        }
        if !self.grip_mult.is_finite() {
            self.grip_mult = default.grip_mult;
        }
        self.mass = self.mass.clamp(MORPHO_MASS_MIN, MORPHO_MASS_MAX);
        self.wheelbase = self
            .wheelbase
            .clamp(MORPHO_WHEELBASE_MIN, MORPHO_WHEELBASE_MAX);
        self.downforce = self
            .downforce
            .clamp(MORPHO_DOWNFORCE_MIN, MORPHO_DOWNFORCE_MAX);
        self.grip_mult = self.grip_mult.clamp(MORPHO_GRIP_MIN, MORPHO_GRIP_MAX);
        self
    }

    /// Mutates each gene independently with probability MORPHO_MUTATION_RATE.
    /// Gaussian noise (Box-Muller) with σ = MORPHO_SIGMA_FRAC × (max - min).
    /// `rng01` must return a uniform f32 in [0, 1).
    pub fn mutate<R: FnMut() -> f32>(&mut self, mut rng01: R) {
        #[inline]
        fn gauss(rng01: &mut impl FnMut() -> f32) -> f32 {
            // Box-Muller with a clamp to avoid ln(0).
            let u1 = rng01().max(1e-7);
            let u2 = rng01();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        }
        let bounds = [
            (MORPHO_MASS_MIN, MORPHO_MASS_MAX),
            (MORPHO_WHEELBASE_MIN, MORPHO_WHEELBASE_MAX),
            (MORPHO_DOWNFORCE_MIN, MORPHO_DOWNFORCE_MAX),
            (MORPHO_GRIP_MIN, MORPHO_GRIP_MAX),
        ];
        for (i, (min, max)) in bounds.into_iter().enumerate() {
            if rng01() < MORPHO_MUTATION_RATE {
                let sigma = (max - min) * MORPHO_SIGMA_FRAC;
                let noise = gauss(&mut rng01) * sigma;
                let val_ref: &mut f32 = match i {
                    0 => &mut self.mass,
                    1 => &mut self.wheelbase,
                    2 => &mut self.downforce,
                    _ => &mut self.grip_mult,
                };
                *val_ref = (*val_ref + noise).clamp(min, max);
            }
        }
    }
}

#[derive(Clone)]
pub struct EliteDna {
    pub fast_weights: Vec<i8>,
    pub fast_scale: f32,
    pub slow_weights: Vec<i8>,
    pub slow_scale: f32,
    pub fitness: f32,
    /// Runtime-only. Length == fast_weights.len(). TTL per weight (0 = free).
    pub scar_mask: Vec<u8>,
    // --- Tier 4.2: Partial epigenetics (runtime-only, not serialized in .dna) ---
    /// Mean cortisol of the organism over its lifetime (trauma baseline).
    pub cortisol_mean: f32,
    /// Mean dopamine (RPE) of the organism (mood baseline).
    pub dopamine_mean: f32,
    /// Mean curiosity of the organism (exploratory tendency).
    pub curiosity_mean: f32,
    // --- Tier 6.3 LITE: morphological co-evolution (serialized in .dna v3.1) ---
    pub morpho: MorphoGenes,
}

/// Marks ~SCAR_FRACTION of the NON-ZERO weights in `fast_weights` with a scar TTL.
/// Deterministic via seed (derived from fast_weights.iter().sum); returns the number of scars created.
pub fn apply_catastrophic_scars(
    scar_mask: &mut Vec<u8>,
    fast_weights: &[i8],
    seed_bias: u64,
) -> usize {
    if scar_mask.len() != fast_weights.len() {
        *scar_mask = vec![0u8; fast_weights.len()];
    }
    // List of non-zero indices
    let nonzero_idx: Vec<usize> = fast_weights
        .iter()
        .enumerate()
        .filter_map(|(i, &w)| if w != 0 { Some(i) } else { None })
        .collect();
    if nonzero_idx.is_empty() {
        return 0;
    }
    let target = ((nonzero_idx.len() as f32) * SCAR_FRACTION).ceil() as usize;
    // Pseudo-random selection via a simple LCG with seed_bias (determinism per DNA).
    let mut state: u64 = seed_bias
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xDEADBEEF);
    let mut scars_created = 0usize;
    for _ in 0..target {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (state >> 32) as usize % nonzero_idx.len();
        let idx = nonzero_idx[pick];
        // Only counts if the weight was not already scarred (a refresh also extends the TTL).
        if scar_mask[idx] == 0 {
            scars_created += 1;
        }
        scar_mask[idx] = SCAR_TTL_INITIAL;
    }
    scars_created
}

/// Decrements the TTLs of every scar by 1 (called at each gen boundary).
/// Returns the number of scars still active after the decrement.
pub fn decay_scars(scar_mask: &mut [u8]) -> usize {
    let mut still_active = 0usize;
    for ttl in scar_mask.iter_mut() {
        if *ttl > 0 {
            *ttl -= 1;
            if *ttl > 0 {
                still_active += 1;
            }
        }
    }
    still_active
}

// ============================================================================
// DNA FILE FORMAT v3 (+ v3.1 morpho tail)
// ----------------------------------------------------------------------------
// Core layout (little-endian, 8478 bytes) + optional morpho tail (+16B = 8494):
//   magic[4]     = b"IARA"
//   version:u16  = 3
//   generation:u32
//   seed:u32
//   track_width:f32
//   fitness:f32
//   fast_scale:f32
//   fast_weights[8192] (i8 as u8)
//   slow_scale:f32
//   slow_weights[256]  (i8 as u8)
//   [TIER 6.3 LITE tail, +16B] morpho: mass:f32 | wheelbase:f32 | downforce:f32 | grip_mult:f32
//
// Backward-compat:
//   - legacy v1 (8196B) and v2 (8456B): no magic, loaded with zeroed metadata.
//   - old v3 (8478B, no morpho tail): MorphoGenes::default().
//   - v3.1 (8494B): decodes the final 16 bytes.
// ============================================================================
pub const DNA_MAGIC: &[u8; 4] = b"IARA";
pub const DNA_VERSION: u16 = 3;
pub const DNA_FAST_SIZE: usize = 128 * 64;
pub const DNA_SLOW_SIZE: usize = 32 * 8;
pub const DNA_HEADER_SIZE: usize = 4 + 2 + 4 + 4 + 4 + 4; // 22
pub const DNA_V3_TOTAL: usize = DNA_HEADER_SIZE + 4 + DNA_FAST_SIZE + 4 + DNA_SLOW_SIZE; // 8478
pub const DNA_V31_TOTAL: usize = DNA_V3_TOTAL + MORPHO_BYTES; // 8494
pub const DNA_V32_TOTAL: usize = DNA_V31_TOTAL + HYPERNET_BYTES; // 8755 (v3.2 with hypernet tail)

// ============================================================================
// TIER 6.4 LITE — HYPERNETWORK GENOME (smallest viable slice)
// ----------------------------------------------------------------------------
// Compress first 2048 weights of instinct_fast as 64 f32 coefficients expanded
// via fixed basis function (sin * coeff * coeff). Opt-in via `active` flag —
// default INACTIVE so existing champions are unaffected.
//
// Serialization tail (after morpho tail, +260 bytes total):
//   1 byte   active flag  (0 = inactive, 1 = active)
//   4 bytes  coeff len LE (always 64, future-proof)
//   256 bytes coeffs     (64 × f32 LE)
//
// Backward-compat: files without this tail default to HypernetSeed::default()
// (inactive). Files with tail decode normally.
// ============================================================================
pub const HYPERNET_COEFFS: usize = 64;
pub const HYPERNET_EXPAND_SIZE: usize = 2048;
pub const HYPERNET_BYTES: usize = 1 + 4 + HYPERNET_COEFFS * 4; // 260
pub const HYPERNET_MUTATION_FLIP_RATE: f32 = 0.02;
pub const HYPERNET_SIGMA: f32 = 0.05;
pub const HYPERNET_TERNARY_THRESH: f32 = 0.3;

#[derive(Clone, Copy, Debug)]
pub struct HypernetSeed {
    pub coeffs: [f32; HYPERNET_COEFFS],
    pub active: bool,
}

impl Default for HypernetSeed {
    fn default() -> Self {
        Self {
            coeffs: [0.0; HYPERNET_COEFFS],
            active: false,
        }
    }
}

impl HypernetSeed {
    pub fn to_le_bytes(&self) -> [u8; HYPERNET_BYTES] {
        let mut out = [0u8; HYPERNET_BYTES];
        out[0] = if self.active { 1 } else { 0 };
        out[1..5].copy_from_slice(&(HYPERNET_COEFFS as u32).to_le_bytes());
        for (i, c) in self.coeffs.iter().enumerate() {
            let off = 5 + i * 4;
            out[off..off + 4].copy_from_slice(&c.to_le_bytes());
        }
        out
    }

    pub fn from_le_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < HYPERNET_BYTES {
            return None;
        }
        let active = buf[0] != 0;
        let len = u32::from_le_bytes(buf[1..5].try_into().ok()?) as usize;
        if len != HYPERNET_COEFFS {
            return None;
        }
        let mut coeffs = [0.0f32; HYPERNET_COEFFS];
        for i in 0..HYPERNET_COEFFS {
            let off = 5 + i * 4;
            let v = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
            coeffs[i] = if v.is_finite() { v } else { 0.0 };
        }
        Some(Self { coeffs, active })
    }

    /// Expands the 64 coeffs into 2048 ternary weights via a fixed sin-product basis.
    pub fn expand(&self) -> [i8; HYPERNET_EXPAND_SIZE] {
        let mut out = [0i8; HYPERNET_EXPAND_SIZE];
        for i in 0..HYPERNET_EXPAND_SIZE {
            let a = self.coeffs[i % HYPERNET_COEFFS];
            let b = self.coeffs[(i.wrapping_mul(7)) % HYPERNET_COEFFS];
            let phase = a * (i as f32 / HYPERNET_EXPAND_SIZE as f32) * std::f32::consts::PI;
            let w = phase.sin() * b;
            out[i] = if w > HYPERNET_TERNARY_THRESH {
                1
            } else if w < -HYPERNET_TERNARY_THRESH {
                -1
            } else {
                0
            };
        }
        out
    }

    /// 2% chance of flipping `active` per gen. If active, Gaussian noise σ=0.05 on the coeffs.
    pub fn mutate<R: FnMut() -> f32>(&mut self, mut rng01: R) {
        #[inline]
        fn gauss(rng01: &mut impl FnMut() -> f32) -> f32 {
            let u1 = rng01().max(1e-7);
            let u2 = rng01();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        }
        if rng01() < HYPERNET_MUTATION_FLIP_RATE {
            self.active = !self.active;
        }
        if self.active {
            for c in self.coeffs.iter_mut() {
                *c += gauss(&mut rng01) * HYPERNET_SIGMA;
                if !c.is_finite() {
                    *c = 0.0;
                }
            }
        }
    }
}

pub struct DnaPayload {
    pub fast_weights: Vec<i8>,
    pub fast_scale: f32,
    pub slow_weights: Option<Vec<i8>>,
    pub slow_scale: f32,
    pub generation: u32,
    pub seed: u32,
    pub track_width: f32,
    pub fitness: f32,
    pub version: u16,
    // --- Tier 6.3 LITE: defaults for old files with no morpho tail ---
    pub morpho: MorphoGenes,
    // --- Tier 6.4 LITE: hypernet seed (inactive default for backward-compat) ---
    pub hypernet: HypernetSeed,
}

pub fn save_champion_dna(
    path: &str,
    fast_weights: &[i8],
    fast_scale: f32,
    slow_weights: &[i8],
    slow_scale: f32,
    generation: u32,
    seed: u32,
    track_width: f32,
    fitness: f32,
) -> std::io::Result<()> {
    save_champion_dna_with_morpho(
        path,
        fast_weights,
        fast_scale,
        slow_weights,
        slow_scale,
        generation,
        seed,
        track_width,
        fitness,
        MorphoGenes::default(),
    )
}

pub fn save_champion_dna_with_morpho(
    path: &str,
    fast_weights: &[i8],
    fast_scale: f32,
    slow_weights: &[i8],
    slow_scale: f32,
    generation: u32,
    seed: u32,
    track_width: f32,
    fitness: f32,
    morpho: MorphoGenes,
) -> std::io::Result<()> {
    save_champion_dna_full(
        path,
        fast_weights,
        fast_scale,
        slow_weights,
        slow_scale,
        generation,
        seed,
        track_width,
        fitness,
        morpho,
        HypernetSeed::default(),
    )
}

pub fn save_champion_dna_full(
    path: &str,
    fast_weights: &[i8],
    fast_scale: f32,
    slow_weights: &[i8],
    slow_scale: f32,
    generation: u32,
    seed: u32,
    track_width: f32,
    fitness: f32,
    morpho: MorphoGenes,
    hypernet: HypernetSeed,
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(DNA_MAGIC)?;
    file.write_all(&DNA_VERSION.to_le_bytes())?;
    file.write_all(&generation.to_le_bytes())?;
    file.write_all(&seed.to_le_bytes())?;
    file.write_all(&track_width.to_le_bytes())?;
    file.write_all(&fitness.to_le_bytes())?;
    file.write_all(&fast_scale.to_le_bytes())?;
    file.write_all(&fast_weights.iter().map(|&w| w as u8).collect::<Vec<u8>>())?;
    file.write_all(&slow_scale.to_le_bytes())?;
    file.write_all(&slow_weights.iter().map(|&w| w as u8).collect::<Vec<u8>>())?;
    // --- Tier 6.3 LITE tail (morpho genes) ---
    file.write_all(&morpho.sanitized().to_le_bytes())?;
    // --- Tier 6.4 LITE tail (hypernet seed) ---
    file.write_all(&hypernet.to_le_bytes())?;
    Ok(())
}

pub fn load_champion_dna(path: &str) -> Option<DnaPayload> {
    let mut file = File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;

    // --- v3 with magic ---
    if buf.len() >= DNA_V3_TOTAL && &buf[0..4] == DNA_MAGIC {
        let version = u16::from_le_bytes(buf[4..6].try_into().ok()?);
        let generation = u32::from_le_bytes(buf[6..10].try_into().ok()?);
        let seed = u32::from_le_bytes(buf[10..14].try_into().ok()?);
        let track_width = f32::from_le_bytes(buf[14..18].try_into().ok()?);
        let fitness = f32::from_le_bytes(buf[18..22].try_into().ok()?);
        let fast_scale_raw = f32::from_le_bytes(buf[22..26].try_into().ok()?);
        let fast_end = 26 + DNA_FAST_SIZE;
        let fast_weights: Vec<i8> = buf[26..fast_end].iter().map(|&b| b as i8).collect();
        let slow_scale_raw = f32::from_le_bytes(buf[fast_end..fast_end + 4].try_into().ok()?);
        let slow_start = fast_end + 4;
        let slow_end = slow_start + DNA_SLOW_SIZE;
        if buf.len() < slow_end {
            return None;
        }
        let slow_weights: Vec<i8> = buf[slow_start..slow_end].iter().map(|&b| b as i8).collect();
        // SANITIZATION: old DNAs may carry astronomical scales (pre-fix myelinate
        // bug with no cap). Clamp to [1e-4, 10] — values outside that indicate
        // runtime corruption, not valid weights. Forces recalibration at the next
        // quantize() via the abs_mean of the ternary weights.
        let fast_scale = if fast_scale_raw.is_finite() {
            fast_scale_raw.clamp(1e-4, 10.0)
        } else {
            1.0
        };
        let slow_scale = if slow_scale_raw.is_finite() {
            slow_scale_raw.clamp(1e-4, 10.0)
        } else {
            1.0
        };
        // --- Tier 6.3 LITE: optional morpho tail (backward-compat with old v3) ---
        let morpho = if buf.len() >= slow_end + MORPHO_BYTES {
            MorphoGenes::from_le_bytes(&buf[slow_end..slow_end + MORPHO_BYTES]).unwrap_or_default()
        } else {
            MorphoGenes::default()
        };
        // --- Tier 6.4 LITE: optional hypernet tail (backward-compat) ---
        let hypernet_start = slow_end + MORPHO_BYTES;
        let hypernet = if buf.len() >= hypernet_start + HYPERNET_BYTES {
            HypernetSeed::from_le_bytes(&buf[hypernet_start..hypernet_start + HYPERNET_BYTES])
                .unwrap_or_default()
        } else {
            HypernetSeed::default()
        };
        return Some(DnaPayload {
            fast_weights,
            fast_scale,
            slow_weights: Some(slow_weights),
            slow_scale,
            generation,
            seed,
            track_width,
            fitness,
            version,
            morpho,
            hypernet,
        });
    }

    // --- Legacy v1/v2 (no magic) ---
    if buf.len() >= 4 + DNA_FAST_SIZE {
        let fast_scale = f32::from_le_bytes(buf[0..4].try_into().ok()?);
        let fast_end = 4 + DNA_FAST_SIZE;
        let fast_weights: Vec<i8> = buf[4..fast_end].iter().map(|&b| b as i8).collect();
        let (slow_weights, slow_scale, version) = if buf.len() >= fast_end + 4 + DNA_SLOW_SIZE {
            let ss = f32::from_le_bytes(buf[fast_end..fast_end + 4].try_into().ok()?);
            let sw: Vec<i8> = buf[fast_end + 4..fast_end + 4 + DNA_SLOW_SIZE]
                .iter()
                .map(|&b| b as i8)
                .collect();
            (Some(sw), ss, 2u16)
        } else {
            (None, 1.0, 1u16)
        };
        return Some(DnaPayload {
            fast_weights,
            fast_scale,
            slow_weights,
            slow_scale,
            generation: 0,
            seed: 0,
            track_width: 0.0,
            fitness: 0.0,
            version,
            morpho: MorphoGenes::default(),
            hypernet: HypernetSeed::default(),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn sample_weights(offset: i8) -> (Vec<i8>, Vec<i8>) {
        let fast: Vec<i8> = (0..DNA_FAST_SIZE)
            .map(|i| (((i as i32 + offset as i32) % 3) - 1) as i8)
            .collect();
        let slow: Vec<i8> = (0..DNA_SLOW_SIZE)
            .map(|i| (((i as i32 + offset as i32) % 3) - 1) as i8)
            .collect();
        (fast, slow)
    }

    #[test]
    fn v3_round_trip_preserves_everything() {
        let (fast, slow) = sample_weights(0);
        let path = std::env::temp_dir().join("iara_test_v3.dna");
        let path_s = path.to_str().unwrap();
        save_champion_dna(path_s, &fast, 0.1234, &slow, 0.5678, 42, 1337, 14.5, 999.9).unwrap();
        let loaded = load_champion_dna(path_s).expect("round trip must succeed");
        assert_eq!(loaded.version, DNA_VERSION);
        assert_eq!(loaded.generation, 42);
        assert_eq!(loaded.seed, 1337);
        assert!((loaded.track_width - 14.5).abs() < 1e-5);
        assert!((loaded.fitness - 999.9).abs() < 1e-3);
        assert!((loaded.fast_scale - 0.1234).abs() < 1e-5);
        assert!((loaded.slow_scale - 0.5678).abs() < 1e-5);
        assert_eq!(loaded.fast_weights, fast);
        assert_eq!(loaded.slow_weights.unwrap(), slow);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn legacy_v2_loads_with_zero_metadata() {
        let (fast, slow) = sample_weights(1);
        let path = std::env::temp_dir().join("iara_test_v2.dna");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0.5_f32.to_le_bytes()).unwrap();
        f.write_all(&fast.iter().map(|&w| w as u8).collect::<Vec<u8>>())
            .unwrap();
        f.write_all(&0.25_f32.to_le_bytes()).unwrap();
        f.write_all(&slow.iter().map(|&w| w as u8).collect::<Vec<u8>>())
            .unwrap();
        drop(f);
        let loaded = load_champion_dna(path.to_str().unwrap()).expect("v2 must load");
        assert_eq!(loaded.version, 2);
        assert_eq!(loaded.generation, 0);
        assert_eq!(loaded.seed, 0);
        assert_eq!(loaded.fast_weights, fast);
        assert_eq!(loaded.slow_weights.unwrap(), slow);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn missing_file_returns_none() {
        let path = std::env::temp_dir().join("iara_does_not_exist.dna");
        let _ = std::fs::remove_file(&path);
        assert!(load_champion_dna(path.to_str().unwrap()).is_none());
    }

    #[test]
    fn v3_total_size_matches_layout() {
        assert_eq!(
            DNA_V3_TOTAL,
            4 + 2 + 4 + 4 + 4 + 4 + 4 + DNA_FAST_SIZE + 4 + DNA_SLOW_SIZE
        );
        assert_eq!(DNA_V31_TOTAL, DNA_V3_TOTAL + MORPHO_BYTES);
    }

    #[test]
    fn morpho_round_trip_and_file_size() {
        let (fast, slow) = sample_weights(2);
        let path = std::env::temp_dir().join("iara_test_v31.dna");
        let path_s = path.to_str().unwrap();
        let morpho = MorphoGenes {
            mass: 1234.5,
            wheelbase: 2.1,
            downforce: 1.7,
            grip_mult: 0.9,
        };
        save_champion_dna_with_morpho(path_s, &fast, 0.1, &slow, 0.2, 1, 2, 10.0, 50.0, morpho)
            .unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len() as usize, DNA_V32_TOTAL);
        let loaded = load_champion_dna(path_s).expect("v3.1 must load");
        assert!((loaded.morpho.mass - 1234.5).abs() < 1e-1);
        assert!((loaded.morpho.wheelbase - 2.1).abs() < 1e-5);
        assert!((loaded.morpho.downforce - 1.7).abs() < 1e-5);
        assert!((loaded.morpho.grip_mult - 0.9).abs() < 1e-5);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn morpho_sanitize_clamps_out_of_range() {
        let wild = MorphoGenes {
            mass: 9000.0,
            wheelbase: 0.1,
            downforce: -5.0,
            grip_mult: 99.0,
        };
        let s = wild.sanitized();
        assert!(s.mass <= MORPHO_MASS_MAX && s.mass >= MORPHO_MASS_MIN);
        assert!(s.wheelbase <= MORPHO_WHEELBASE_MAX && s.wheelbase >= MORPHO_WHEELBASE_MIN);
        assert!(s.downforce <= MORPHO_DOWNFORCE_MAX && s.downforce >= MORPHO_DOWNFORCE_MIN);
        assert!(s.grip_mult <= MORPHO_GRIP_MAX && s.grip_mult >= MORPHO_GRIP_MIN);
    }

    #[test]
    fn legacy_v3_without_morpho_loads_with_default() {
        // Simulates a v3 file saved pre-tier-6.3 (8478 bytes, no morpho tail).
        let (fast, slow) = sample_weights(3);
        let path = std::env::temp_dir().join("iara_test_v3_legacy.dna");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(DNA_MAGIC).unwrap();
        f.write_all(&DNA_VERSION.to_le_bytes()).unwrap();
        f.write_all(&7_u32.to_le_bytes()).unwrap();
        f.write_all(&123_u32.to_le_bytes()).unwrap();
        f.write_all(&12.0_f32.to_le_bytes()).unwrap();
        f.write_all(&42.0_f32.to_le_bytes()).unwrap();
        f.write_all(&0.3_f32.to_le_bytes()).unwrap();
        f.write_all(&fast.iter().map(|&w| w as u8).collect::<Vec<u8>>())
            .unwrap();
        f.write_all(&0.4_f32.to_le_bytes()).unwrap();
        f.write_all(&slow.iter().map(|&w| w as u8).collect::<Vec<u8>>())
            .unwrap();
        drop(f);
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len() as usize, DNA_V3_TOTAL);
        let loaded = load_champion_dna(path.to_str().unwrap()).expect("legacy v3 must load");
        let d = MorphoGenes::default();
        assert!((loaded.morpho.mass - d.mass).abs() < 1e-3);
        assert!((loaded.morpho.wheelbase - d.wheelbase).abs() < 1e-5);
        let _ = std::fs::remove_file(path);
    }

    // ────────────────────────────────────────────────────────────────────────
    // CROSS-MODULE EQUIVALENCE TEST (CRITICAL SAFETY NET)
    // ────────────────────────────────────────────────────────────────────────
    // Guarantees that a file saved by dna.rs::save_champion_dna is readable
    // by dna_genesis::load_champion_dna (the loader active in production).
    //
    // The dna.rs vs dna_genesis.rs duplication is DANGEROUS: if the layouts
    // diverge, saving through the wrong path corrupts DNAs in silence.
    //
    // WHY THIS DIVERGENCE EXISTS (do not remove without reading):
    //   - dna.rs has save_champion_dna_full() which writes the flat V3 format.
    //   - dna_genesis.rs has save_champion_dna_full() which delegates to V4 TLV
    //     via compress_organism() (iara_genesis).
    //   - dna.rs::save produces a V3 file (8494-8755 bytes, magic+flat).
    //   - dna_genesis::load reads V3 as the legacy path (version != 4) — works.
    //   - dna_genesis::save_champion_dna_v7 produces a V8 file (TLV iara_genesis).
    //   - dna.rs::load does NOT know the V8 TLV format — it is the OLD loader.
    //
    // SOURCE OF TRUTH: dna_genesis.rs is the only active implementation.
    // dna.rs must be kept ONLY for backward-compat reading of legacy
    // V1/V2/V3 files that exist on disk.
    //
    // Do NOT remove dna.rs without guaranteeing that the V3/V1/V2 legacy path
    // is still in dna_genesis.rs (it already is — see load_champion_dna there).
    // ────────────────────────────────────────────────────────────────────────

    /// 1. dna.rs saves a V3; dna_genesis loads it — weights must match.
    #[test]
    fn dna_legacy_writer_readable_by_genesis_loader() {
        let (fast, slow) = sample_weights(7);
        let path = std::env::temp_dir().join("iara_cross_module_legacy.dna");
        let path_s = path.to_str().unwrap();

        // Saves through the LEGACY path (dna.rs) — produces flat V3.
        save_champion_dna(path_s, &fast, 0.25, &slow, 0.125, 10, 42, 14.0, 123.4)
            .expect("dna.rs::save must work");

        // Loads through the ACTIVE path (dna_genesis) — must be able to read the V3.
        let loaded = crate::evolution::dna_genesis::load_champion_dna(path_s)
            .expect("dna_genesis::load must read a V3 file saved by dna.rs");

        // Metadata must be preserved.
        assert_eq!(loaded.generation, 10, "generation diverged between modules");
        assert_eq!(loaded.seed, 42, "seed diverged between modules");
        assert!(
            (loaded.track_width - 14.0).abs() < 1e-4,
            "track_width diverged"
        );
        assert!((loaded.fitness - 123.4).abs() < 1e-2, "fitness diverged");
        assert!(
            (loaded.fast_scale - 0.25).abs() < 1e-4,
            "fast_scale diverged"
        );
        assert!(
            (loaded.slow_scale - 0.125).abs() < 1e-4,
            "slow_scale diverged"
        );
        assert_eq!(
            loaded.fast_weights, fast,
            "fast_weights diverged between modules"
        );
        assert_eq!(
            loaded.slow_weights.as_deref().unwrap(),
            slow.as_slice(),
            "slow_weights diverged between modules"
        );

        let _ = std::fs::remove_file(path);
    }

    /// 2. DNA size constants in dna.rs must match the LEGACY ones in
    ///    dna_genesis.rs (if they diverge, the legacy parser breaks).
    #[test]
    fn dna_size_constants_match_genesis_legacy() {
        // dna.rs defines DNA_FAST_SIZE = 128*64 = 8192 (legacy).
        // dna_genesis.rs defines DNA_FAST_SIZE_LEGACY = 128*64 = 8192.
        // If someone changes one and not the other, the flat V3 parser reads wrong bytes.
        assert_eq!(
            DNA_FAST_SIZE,
            crate::config::iara::genetics::DNA_FAST_SIZE_LEGACY,
            "DNA_FAST_SIZE in dna.rs != DNA_FAST_SIZE_LEGACY in config — \
             this would break the legacy V3 parser"
        );
        assert_eq!(
            DNA_SLOW_SIZE,
            crate::config::iara::genetics::DNA_SLOW_SIZE_LEGACY,
            "DNA_SLOW_SIZE in dna.rs != DNA_SLOW_SIZE_LEGACY in config — \
             this would break the legacy V3 parser"
        );
    }

    /// 3. MorphoGenes::to_le_bytes / from_le_bytes in dna.rs must be
    ///    bit-for-bit identical to dna_genesis.rs's.
    #[test]
    fn morpho_bytes_identical_across_modules() {
        let m_dna = MorphoGenes {
            mass: 1100.0,
            wheelbase: 2.3,
            downforce: 1.5,
            grip_mult: 1.0,
        };

        // Serializes via dna.rs.
        let bytes_dna = m_dna.to_le_bytes();

        // Deserializes via dna_genesis.rs.
        let m_genesis = crate::evolution::dna_genesis::MorphoGenes::from_le_bytes(&bytes_dna)
            .expect("dna_genesis must deserialize bytes from dna.rs");

        assert!(
            (m_dna.mass - m_genesis.mass).abs() < 1e-4,
            "MorphoGenes.mass diverges: {} vs {}",
            m_dna.mass,
            m_genesis.mass
        );
        assert!(
            (m_dna.wheelbase - m_genesis.wheelbase).abs() < 1e-5,
            "MorphoGenes.wheelbase diverges"
        );
        assert!(
            (m_dna.downforce - m_genesis.downforce).abs() < 1e-5,
            "MorphoGenes.downforce diverges"
        );
        assert!(
            (m_dna.grip_mult - m_genesis.grip_mult).abs() < 1e-5,
            "MorphoGenes.grip_mult diverges"
        );

        // And vice versa: serialize via dna_genesis, deserialize via dna.
        let m_gen2 = crate::evolution::dna_genesis::MorphoGenes {
            mass: 900.0,
            wheelbase: 2.8,
            downforce: 0.7,
            grip_mult: 1.1,
        };
        let bytes_gen2 = m_gen2.to_le_bytes();
        let m_back = MorphoGenes::from_le_bytes(&bytes_gen2)
            .expect("dna.rs must deserialize bytes from dna_genesis");
        assert!(
            (m_gen2.mass - m_back.mass).abs() < 1e-4,
            "inverted round-trip failed"
        );
    }

    /// 4. HypernetSeed::to_le_bytes / from_le_bytes must be identical.
    #[test]
    fn hypernet_bytes_identical_across_modules() {
        let mut h_dna = HypernetSeed::default();
        h_dna.active = true;
        h_dna.coeffs[0] = 0.5;
        h_dna.coeffs[1] = -0.3;

        // dna.rs → dna_genesis.
        let bytes = h_dna.to_le_bytes();
        let h_gen = crate::evolution::dna_genesis::HypernetSeed::from_le_bytes(&bytes)
            .expect("dna_genesis must read a HypernetSeed from dna.rs");
        assert_eq!(h_gen.active, h_dna.active);
        assert!((h_gen.coeffs[0] - h_dna.coeffs[0]).abs() < 1e-5);
        assert!((h_gen.coeffs[1] - h_dna.coeffs[1]).abs() < 1e-5);
    }
}
