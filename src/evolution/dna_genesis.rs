use std::fs::File;
use std::io::{Read, Write};

use iara_genesis::tlv::{compress_organism, decompress_organism, Chunk, GenesisPayload};
use iara_genesis::model::tree::Sap;

// ============================================================================
// TIER 6.6 — WEIGHT SCARS (trauma imutabilidade)
// ----------------------------------------------------------------------------
// Quando um organismo morre em CATÁSTROFE (Batida frontal com HP→0), uma fração
// dos pesos NÃO-ZEROS (proxy de "pesos ativos naquele momento") é marcada como
// "scar tissue" — bloqueada para mutação por N gerações.
//
// - `scar_mask[i]` = TTL u8: 0 = livre, >0 = travado (decrementa a cada gen).
// - Scars sobrevivem transição de geração via EliteDna.
// - Não entra no .dna v3 (runtime-only por enquanto). Documentado p/ v4.
//
// Versão simplificada (v1): snapshot proxy — ativos ≈ não-zeros. Versão completa
// (v2, futura) usaria last_activation_mask do BitLinear após um forward real.
// ============================================================================
pub const SCAR_TTL_INITIAL: u8 = 5;       // trava por 5 gerações
pub const SCAR_FRACTION: f32 = 0.10;      // 10% dos pesos não-zero viram scar
pub const SCAR_CATASTROPHE_LABEL: &str = "BATIDA";

// ============================================================================
// TIER 6.3 LITE — MORPHOLOGICAL CO-EVOLUTION (genes corporais)
// ----------------------------------------------------------------------------
// 4 parâmetros de carroceria co-evoluem com o cérebro. Por ora só são
// ARMAZENADOS (no DNA e no Racer) e mutacionam junto das elites; o wiring
// físico fica pra follow-up.
//   - mass      (kg)     [700, 1400]  default 1000
//   - wheelbase (m)      [1.5, 3.0]   default 2.5
//   - downforce (coef)   [0.5, 2.0]   default 1.0
//   - grip_mult (coef)   [0.8, 1.2]   default 1.0
// Mutação: 5% de chance por gen por gene, ruído gaussiano σ = 5% do range,
// clamp nos limites.
// ============================================================================
pub const MORPHO_MASS_MIN: f32 = 700.0;
pub const MORPHO_MASS_MAX: f32 = 1400.0;
pub const MORPHO_WHEELBASE_MIN: f32 = 1.5;
pub const MORPHO_WHEELBASE_MAX: f32 = 3.0;
pub const MORPHO_DOWNFORCE_MIN: f32 = 0.5;
pub const MORPHO_DOWNFORCE_MAX: f32 = 2.0;
pub const MORPHO_GRIP_MIN: f32 = 0.8;
pub const MORPHO_GRIP_MAX: f32 = 1.2;
pub const MORPHO_MUTATION_RATE: f32 = 0.05; // 5% por gene por gen
pub const MORPHO_SIGMA_FRAC: f32 = 0.05;    // σ = 5% do range
pub const MORPHO_BYTES: usize = 16;         // 4 × f32 LE

#[derive(Clone, Copy, Debug)]
pub struct MorphoGenes {
    pub mass: f32,
    pub wheelbase: f32,
    pub downforce: f32,
    pub grip_mult: f32,
}

impl Default for MorphoGenes {
    fn default() -> Self {
        Self { mass: 1000.0, wheelbase: 2.5, downforce: 1.0, grip_mult: 1.0 }
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
        if buf.len() < MORPHO_BYTES { return None; }
        let mass = f32::from_le_bytes(buf[0..4].try_into().ok()?);
        let wheelbase = f32::from_le_bytes(buf[4..8].try_into().ok()?);
        let downforce = f32::from_le_bytes(buf[8..12].try_into().ok()?);
        let grip_mult = f32::from_le_bytes(buf[12..16].try_into().ok()?);
        let out = Self { mass, wheelbase, downforce, grip_mult };
        Some(out.sanitized())
    }

    /// Clamp nos ranges (+ proteção contra NaN/Inf vinda de arquivo corrompido).
    pub fn sanitized(mut self) -> Self {
        let default = Self::default();
        if !self.mass.is_finite()      { self.mass = default.mass; }
        if !self.wheelbase.is_finite() { self.wheelbase = default.wheelbase; }
        if !self.downforce.is_finite() { self.downforce = default.downforce; }
        if !self.grip_mult.is_finite() { self.grip_mult = default.grip_mult; }
        self.mass      = self.mass.clamp(MORPHO_MASS_MIN, MORPHO_MASS_MAX);
        self.wheelbase = self.wheelbase.clamp(MORPHO_WHEELBASE_MIN, MORPHO_WHEELBASE_MAX);
        self.downforce = self.downforce.clamp(MORPHO_DOWNFORCE_MIN, MORPHO_DOWNFORCE_MAX);
        self.grip_mult = self.grip_mult.clamp(MORPHO_GRIP_MIN, MORPHO_GRIP_MAX);
        self
    }

    /// Muta cada gene independentemente com probabilidade MORPHO_MUTATION_RATE.
    /// Ruído gaussiano (Box-Muller) com σ = MORPHO_SIGMA_FRAC × (max - min).
    /// `rng01` deve retornar f32 uniforme em [0, 1).
    pub fn mutate<R: FnMut() -> f32>(&mut self, mut rng01: R) {
        #[inline]
        fn gauss(rng01: &mut impl FnMut() -> f32) -> f32 {
            // Box-Muller com clamp pra evitar ln(0).
            let u1 = rng01().max(1e-7);
            let u2 = rng01();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        }
        let bounds = [
            (MORPHO_MASS_MIN,      MORPHO_MASS_MAX),
            (MORPHO_WHEELBASE_MIN, MORPHO_WHEELBASE_MAX),
            (MORPHO_DOWNFORCE_MIN, MORPHO_DOWNFORCE_MAX),
            (MORPHO_GRIP_MIN,      MORPHO_GRIP_MAX),
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
    pub fast_weights: Vec<f32>, // f32 runtime; quantize só ao salvar DNA
    pub fast_scale: f32,
    pub slow_weights: Vec<f32>,
    pub slow_scale: f32,
    pub cortex_l2: Vec<f32>,
    pub cortex_l3: Vec<f32>,
    pub fitness: f32,
    /// Runtime-only. Comprimento == fast_weights.len(). TTL por peso (0 = livre).
    pub scar_mask: Vec<u8>,
    // --- Tier 4.2: Epigenética parcial (runtime-only, não serializa em .dna) ---
    /// Média de cortisol do organismo ao longo da sua vida (trauma baseline).
    pub cortisol_mean: f32,
    /// Média de dopamina (RPE) do organismo (mood baseline).
    pub dopamine_mean: f32,
    /// Média de curiosidade do organismo (tendência exploratória).
    pub curiosity_mean: f32,
    // --- Tier 6.3 LITE: morphological co-evolution (serializado em .dna v3.1) ---
    pub morpho: MorphoGenes,
    pub car_model_idx: u8,
    /// v11: KAN output head knots (runtime-only, não serializa em .dna).
    /// 3 outputs × KAN_OUT_KNOTS cada. Preserva spline aprendido entre gens.
    pub kan_knots: [[f32; crate::kan::KAN_OUT_KNOTS]; 3],
}

/// Marca ~SCAR_FRACTION dos pesos NÃO-ZERO em `fast_weights` com scar TTL.
/// Determinístico via seed (deriva de fast_weights.iter().sum); retorna nº de scars criados.
pub fn apply_catastrophic_scars(scar_mask: &mut Vec<u8>, fast_weights: &[i8], seed_bias: u64) -> usize {
    if scar_mask.len() != fast_weights.len() {
        *scar_mask = vec![0u8; fast_weights.len()];
    }
    // Lista de índices não-zero
    let nonzero_idx: Vec<usize> = fast_weights.iter().enumerate()
        .filter_map(|(i, &w)| if w != 0 { Some(i) } else { None })
        .collect();
    if nonzero_idx.is_empty() { return 0; }
    let target = ((nonzero_idx.len() as f32) * SCAR_FRACTION).ceil() as usize;
    // Seleção pseudo-aleatória via LCG simples com seed_bias (determinismo por DNA).
    let mut state: u64 = seed_bias.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xDEADBEEF);
    let mut scars_created = 0usize;
    for _ in 0..target {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let pick = (state >> 32) as usize % nonzero_idx.len();
        let idx = nonzero_idx[pick];
        // Só conta se o peso não estava já scarred (refresh também estende TTL).
        if scar_mask[idx] == 0 { scars_created += 1; }
        scar_mask[idx] = SCAR_TTL_INITIAL;
    }
    scars_created
}

/// Decrementa TTLs de todos os scars em 1 (chamado a cada gen boundary).
/// Retorna nº de scars ainda ativos após decremento.
pub fn decay_scars(scar_mask: &mut [u8]) -> usize {
    let mut still_active = 0usize;
    for ttl in scar_mask.iter_mut() {
        if *ttl > 0 {
            *ttl -= 1;
            if *ttl > 0 { still_active += 1; }
        }
    }
    still_active
}

// ============================================================================
// DNA FILE FORMAT v3 (+ v3.1 morpho tail)
// ----------------------------------------------------------------------------
// Layout core (little-endian, 8478 bytes) + opcional tail morpho (+16B = 8494):
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
//   - legacy v1 (8196B) e v2 (8456B): sem magic, carregados com metadata zerada.
//   - v3 antigo (8478B, sem tail morpho): MorphoGenes::default().
//   - v3.1 (8494B): decodifica os 16 bytes finais.
// ============================================================================
pub const DNA_MAGIC: &[u8; 4] = b"IARA";
pub use crate::config::iara::genetics::*;



pub const DNA_HEADER_SIZE: usize = 4 + 2 + 4 + 4 + 4 + 4;  // 22



pub const DNA_V3_TOTAL: usize = DNA_HEADER_SIZE + 4 + DNA_FAST_SIZE_LEGACY + 4 + DNA_SLOW_SIZE_LEGACY;  // 8478
pub const DNA_V31_TOTAL: usize = DNA_V3_TOTAL + MORPHO_BYTES;  // 8494
pub const DNA_V32_TOTAL: usize = DNA_V31_TOTAL + HYPERNET_BYTES; // 8755 (v3.2 with hypernet tail)

// TLV TAG para metadados exclusivos do simulador (usado apenas na V4)
pub const TAG_META_UNIVERSE: &[u8; 4] = b"META";
// TLV TAG para Trinity weights (gate + reflex). Opcional — DNAs antigos não têm.
pub const TAG_TRINITY: &[u8; 4] = b"TRIN";
// TLV TAG para cortex equation (DCT compression). Opcional — substitui fast_weights raw.
pub const TAG_CORTEX_EQ: &[u8; 4] = b"CEQU";
// TLV TAG para slow_brain weights (V6 equation-only DNA — sem TREE/GENE).
pub const TAG_SLOW_RAW: &[u8; 4] = b"SLOW";
// TLV TAG para slow_brain weights f32 RAW (bypass Tree/sparse::compress que
// clamp não-ternários a zero). 256 weights × 4 bytes = 1024 bytes. Lossless f32.
pub const TAG_SLOW_F32: &[u8; 4] = b"SLWF";
// TLV TAG para cortex weights raw i8 (bypass compress_organism que quantiza ternary).
// Necessário pra preservar weights f32 trained com fidelidade plena (i8/63.5 round-trip).
pub const TAG_CORTEX_RAW: &[u8; 4] = b"CRAW";
// TLV TAG para cortex_l2 (Layer 2 deep composition, 64→32 = 2048 weights).
pub const TAG_CORTEX_L2: &[u8; 4] = b"CL2W";

/// Cortex DCT compression — 256 coeffs cobrem 8192 weights via inverse DCT-II.
/// 32× compressão. Lossy mas smooth. Pra DNAs onde fidelidade absoluta não é
/// crítica (cars treinados convergem mesmo com aproximação).
pub const CORTEX_DCT_COEFFS: usize = 256;
pub const CORTEX_FULL_SIZE: usize = DNA_FAST_SIZE; // Retina HD v2
pub const CORTEX_DCT_BYTES: usize = CORTEX_DCT_COEFFS * 4; // 1024 bytes

/// Forward DCT-II truncated to K coefficients.
/// X[k] = (2/N) Σ x[n] cos(π/N (n+0.5) k)
pub fn cortex_to_dct(weights: &[f32]) -> [f32; CORTEX_DCT_COEFFS] {
    let n = weights.len().min(CORTEX_FULL_SIZE);
    let mut out = [0.0_f32; CORTEX_DCT_COEFFS];
    let n_inv = 1.0 / n as f32;
    let pi_n = std::f32::consts::PI * n_inv;
    for k in 0..CORTEX_DCT_COEFFS {
        let mut acc = 0.0_f32;
        for i in 0..n {
            acc += weights[i] * (pi_n * (i as f32 + 0.5) * k as f32).cos();
        }
        out[k] = acc * 2.0 * n_inv;
    }
    out
}

/// Inverse DCT-II reconstructing N weights from K coefficients.
/// x[n] = X[0]/2 + Σ_{k=1..K} X[k] cos(π/N (n+0.5) k)
pub fn cortex_from_dct(coeffs: &[f32; CORTEX_DCT_COEFFS]) -> Vec<f32> {
    let n = CORTEX_FULL_SIZE;
    let mut out = vec![0.0_f32; n];
    let pi_n = std::f32::consts::PI / n as f32;
    for i in 0..n {
        let mut acc = coeffs[0] * 0.5;
        for k in 1..CORTEX_DCT_COEFFS {
            acc += coeffs[k] * (pi_n * (i as f32 + 0.5) * k as f32).cos();
        }
        out[i] = acc;
    }
    out
}

pub fn dct_coeffs_to_bytes(coeffs: &[f32; CORTEX_DCT_COEFFS]) -> [u8; CORTEX_DCT_BYTES] {
    let mut out = [0u8; CORTEX_DCT_BYTES];
    for (i, c) in coeffs.iter().enumerate() {
        out[i*4..i*4+4].copy_from_slice(&c.to_le_bytes());
    }
    out
}

pub fn dct_coeffs_from_bytes(buf: &[u8]) -> Option<[f32; CORTEX_DCT_COEFFS]> {
    if buf.len() < CORTEX_DCT_BYTES { return None; }
    let mut out = [0.0_f32; CORTEX_DCT_COEFFS];
    for i in 0..CORTEX_DCT_COEFFS {
        let v = f32::from_le_bytes(buf[i*4..i*4+4].try_into().ok()?);
        out[i] = if v.is_finite() { v } else { 0.0 };
    }
    Some(out)
}

// Trinity payload: gate weights (16*3=48 f32) + gate bias (3 f32) + reflex i8 (32*3=96)
pub const TRINITY_GATE_FLOATS: usize = 16 * 3 + 3; // 51
pub const TRINITY_REFLEX_BYTES: usize = 32 * 3;    // 96
pub const TRINITY_BYTES: usize = TRINITY_GATE_FLOATS * 4 + TRINITY_REFLEX_BYTES; // 300

#[derive(Clone, Debug)]
pub struct TrinityWeights {
    pub gate_weights: [f32; 48],
    pub gate_bias: [f32; 3],
    pub reflex_steer: [i8; 32],
    pub reflex_throttle: [i8; 32],
    pub reflex_brake: [i8; 32],
}

impl TrinityWeights {
    pub fn to_bytes(&self) -> [u8; TRINITY_BYTES] {
        let mut out = [0u8; TRINITY_BYTES];
        let mut o = 0;
        for &w in &self.gate_weights { out[o..o+4].copy_from_slice(&w.to_le_bytes()); o += 4; }
        for &b in &self.gate_bias    { out[o..o+4].copy_from_slice(&b.to_le_bytes()); o += 4; }
        for &w in &self.reflex_steer    { out[o] = w as u8; o += 1; }
        for &w in &self.reflex_throttle { out[o] = w as u8; o += 1; }
        for &w in &self.reflex_brake    { out[o] = w as u8; o += 1; }
        out
    }
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < TRINITY_BYTES { return None; }
        let mut gate_weights = [0.0f32; 48];
        let mut gate_bias = [0.0f32; 3];
        let mut reflex_steer = [0i8; 32];
        let mut reflex_throttle = [0i8; 32];
        let mut reflex_brake = [0i8; 32];
        let mut o = 0;
        for w in gate_weights.iter_mut() { *w = f32::from_le_bytes(buf[o..o+4].try_into().ok()?); o += 4; }
        for b in gate_bias.iter_mut()    { *b = f32::from_le_bytes(buf[o..o+4].try_into().ok()?); o += 4; }
        for w in reflex_steer.iter_mut()    { *w = buf[o] as i8; o += 1; }
        for w in reflex_throttle.iter_mut() { *w = buf[o] as i8; o += 1; }
        for w in reflex_brake.iter_mut()    { *w = buf[o] as i8; o += 1; }
        Some(Self { gate_weights, gate_bias, reflex_steer, reflex_throttle, reflex_brake })
    }
}


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

// ============================================================================
// KAN OUTPUT-HEAD SPLINE KNOTS (v3.3 / V4 META tail)
// ----------------------------------------------------------------------------
// 3 outputs (throttle/steer/brake) × KAN_OUT_KNOTS f32 cada. Preserva spline
// aprendido entre gens — antes era re-iniciado a cada load. Serializado como
// tail do chunk META (V4) ou tail de arquivo (V3 legacy load).
//
// Layout: 144 bytes raw little-endian f32s na ordem [out0_k0..out0_kN, out1_..., out2_...].
// Backward-compat: arquivos sem tail → KanOutputHead::new() default (tanh ramp).
// ============================================================================
pub const KAN_KNOTS_OUTS: usize = 3;
pub const KAN_KNOTS_PER_OUT: usize = crate::kan::KAN_OUT_KNOTS;
pub const KAN_KNOTS_BYTES: usize = KAN_KNOTS_OUTS * KAN_KNOTS_PER_OUT * 4;

pub fn default_kan_knots() -> [[f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS] {
    crate::kan::KanOutputHead::new().knots
}

pub fn kan_knots_to_bytes(knots: &[[f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS]) -> [u8; KAN_KNOTS_BYTES] {
    let mut out = [0u8; KAN_KNOTS_BYTES];
    for o in 0..KAN_KNOTS_OUTS {
        for k in 0..KAN_KNOTS_PER_OUT {
            let off = (o * KAN_KNOTS_PER_OUT + k) * 4;
            out[off..off + 4].copy_from_slice(&knots[o][k].to_le_bytes());
        }
    }
    out
}

pub fn kan_knots_from_bytes(buf: &[u8]) -> Option<[[f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS]> {
    if buf.len() < KAN_KNOTS_BYTES { return None; }
    let mut knots = [[0.0f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS];
    for o in 0..KAN_KNOTS_OUTS {
        for k in 0..KAN_KNOTS_PER_OUT {
            let off = (o * KAN_KNOTS_PER_OUT + k) * 4;
            let v = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
            knots[o][k] = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
        }
    }
    Some(knots)
}

#[derive(Clone, Copy, Debug)]
pub struct HypernetSeed {
    pub coeffs: [f32; HYPERNET_COEFFS],
    pub active: bool,
}

impl Default for HypernetSeed {
    fn default() -> Self {
        Self { coeffs: [0.0; HYPERNET_COEFFS], active: false }
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
        if buf.len() < HYPERNET_BYTES { return None; }
        let active = buf[0] != 0;
        let len = u32::from_le_bytes(buf[1..5].try_into().ok()?) as usize;
        if len != HYPERNET_COEFFS { return None; }
        let mut coeffs = [0.0f32; HYPERNET_COEFFS];
        for i in 0..HYPERNET_COEFFS {
            let off = 5 + i * 4;
            let v = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
            coeffs[i] = if v.is_finite() { v } else { 0.0 };
        }
        Some(Self { coeffs, active })
    }

    /// Expande os 64 coeffs em 2048 pesos ternários via basis sin-product fixa.
    pub fn expand(&self) -> [i8; HYPERNET_EXPAND_SIZE] {
        let mut out = [0i8; HYPERNET_EXPAND_SIZE];
        for i in 0..HYPERNET_EXPAND_SIZE {
            let a = self.coeffs[i % HYPERNET_COEFFS];
            let b = self.coeffs[(i.wrapping_mul(7)) % HYPERNET_COEFFS];
            let phase = a * (i as f32 / HYPERNET_EXPAND_SIZE as f32) * std::f32::consts::PI;
            let w = phase.sin() * b;
            out[i] = if w > HYPERNET_TERNARY_THRESH { 1 }
                     else if w < -HYPERNET_TERNARY_THRESH { -1 }
                     else { 0 };
        }
        out
    }

    /// 2% chance de flip do active por gen. Se ativo, ruído gaussiano σ=0.05 nos coeffs.
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
                if !c.is_finite() { *c = 0.0; }
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
    // --- Tier 6.3 LITE: defaults pra arquivos antigos sem tail morpho ---
    pub morpho: MorphoGenes,
    pub car_model_idx: u8,
    // --- Tier 6.4 LITE: hypernet seed (inactive default p/ backward-compat) ---
    pub hypernet: HypernetSeed,
    // --- v3.3 / V4 META tail: KAN output-head knots (preserva spline aprendido) ---
    pub kan_knots: [[f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS],
    // --- v5 / TLV TRIN chunk: Trinity gate + reflex weights (None se DNA pré-v5) ---
    pub trinity: Option<TrinityWeights>,
    // --- v6 / TLV CEQU chunk: cortex DCT compression (256 coeffs → 8192 weights).
    // Experimental — ainda lossy demais pra ser source-of-truth.
    pub cortex_dct: Option<[f32; CORTEX_DCT_COEFFS]>,
    // --- v6 / TLV CRAW chunk: cortex raw i8 weights (bypass compress_organism
    // que quantizava ternary e destruía f32 trained). Source of truth pra cortex.
    pub cortex_raw: Option<Vec<i8>>,
    // --- v7 / TLV CL2W chunk: cortex layer 2 weights (64×32=2048 i8 quantizado 63.5).
    pub cortex_l2_raw: Option<Vec<i8>>,
    pub cortex_l3_raw: Option<Vec<i8>>,
    // SLWF chunk: slow_brain raw f32 (lossless). Bypassa Tree/sparse compressor ternary.
    pub slow_f32: Option<Vec<f32>>,
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
        path, fast_weights, fast_scale, slow_weights, slow_scale,
        generation, seed, track_width, fitness, MorphoGenes::default(),
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
        path, fast_weights, fast_scale, slow_weights, slow_scale,
        generation, seed, track_width, fitness, morpho, HypernetSeed::default(),
        default_kan_knots(),
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
    kan_knots: [[f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS],
) -> std::io::Result<()> {
    save_champion_dna_v5(
        path, fast_weights, fast_scale, slow_weights, slow_scale,
        generation, seed, track_width, fitness, morpho, hypernet, kan_knots, None,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn save_champion_dna_v5(
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
    kan_knots: [[f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS],
    trinity: Option<&TrinityWeights>,
) -> std::io::Result<()> {
    save_champion_dna_v7(path, fast_weights, fast_scale, slow_weights, slow_scale,
        generation, seed, track_width, fitness, morpho, hypernet, kan_knots, trinity, None, None)
}

#[allow(clippy::too_many_arguments)]
pub fn save_champion_dna_v7(
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
    kan_knots: [[f32; crate::kan::KAN_OUT_KNOTS]; KAN_KNOTS_OUTS],
    trinity: Option<&TrinityWeights>,
    cortex_l2: Option<&[i8]>,
    slow_f32: Option<&[f32]>,
) -> std::io::Result<()> {
    // 1. Gera o payload compressando os genomas (V4)
    let mut payload = compress_organism(
        fast_weights,
        slow_weights,
        generation,
        fitness,
        Sap::default()
    );

    // 2. Cria o chunk META exclusivo do Universo para conter seed, scales, track e morpho
    // Tamanho: seed(4) + t_width(4) + f_scale(4) + s_scale(4) + morpho(16) + hypernet(260) = 292
    let mut meta_data = Vec::with_capacity(300);
    meta_data.extend_from_slice(&seed.to_le_bytes());
    meta_data.extend_from_slice(&track_width.to_le_bytes());
    meta_data.extend_from_slice(&fast_scale.to_le_bytes());
    meta_data.extend_from_slice(&slow_scale.to_le_bytes());
    meta_data.extend_from_slice(&morpho.sanitized().to_le_bytes());
    meta_data.extend_from_slice(&hypernet.to_le_bytes());
    meta_data.extend_from_slice(&kan_knots_to_bytes(&kan_knots));

    payload.add_chunk(Chunk::new(*TAG_META_UNIVERSE, meta_data));

    // Trinity chunk (opcional) — gate + reflex weights pra restaurar swarm aprendido.
    if let Some(t) = trinity {
        payload.add_chunk(Chunk::new(*TAG_TRINITY, t.to_bytes().to_vec()));
    }

    // CRAW chunk (cortex raw i8) — bypass compress_organism's ternary quantization
    // que destruía f32 weights. Esta é a fonte de verdade do cortex L1.
    payload.add_chunk(Chunk::new(*TAG_CORTEX_RAW, fast_weights.iter().map(|&b| b as u8).collect()));

    // CL2W chunk (cortex layer 2, NOVO V7).
    if let Some(l2) = cortex_l2 {
        payload.add_chunk(Chunk::new(*TAG_CORTEX_L2, l2.iter().map(|&b| b as u8).collect()));
    }

    // SLWF chunk (slow_brain raw f32) — bypass Tree/sparse::compress ternary clamp.
    // Sem isso, slow weights = qualquer valor não {-1,0,+1} viravam zero no save.
    if let Some(sw) = slow_f32 {
        let mut bytes = Vec::with_capacity(sw.len() * 4);
        for &w in sw {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        payload.add_chunk(Chunk::new(*TAG_SLOW_F32, bytes));
    }

    // CEQU chunk (cortex equation): DCT compression for future equation-only DNA.
    // Currently EXPERIMENTAL — not used on load (lossy 32× reconstruction não preserva
    // trained net fidelity). Mantido pra iteração futura com basis menos perdida.
    {
        let cortex_f32: Vec<f32> = fast_weights.iter().map(|&w| w as f32 / 63.5).collect();
        let dct = cortex_to_dct(&cortex_f32);
        payload.add_chunk(Chunk::new(*TAG_CORTEX_EQ, dct_coeffs_to_bytes(&dct).to_vec()));
    }

    // 3. Escreve no disco (Header V4 minimalista)
    let mut file = File::create(path)?;
    file.write_all(DNA_MAGIC)?; // "IARA"
    file.write_all(&DNA_VERSION.to_le_bytes())?; // 4
    
    // Escreve todo o envelope generativo TLV
    file.write_all(&payload.to_bytes())?;

    Ok(())
}

pub fn load_champion_dna(path: &str) -> Option<DnaPayload> {
    let mut file = File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;

    // --- v4 with GenesisPayload ---
    if buf.len() > 6 && &buf[0..4] == DNA_MAGIC {
        let version = u16::from_le_bytes(buf[4..6].try_into().ok()?);
        if version == 4 {
            // Desserializa o payload TLV a partir do byte 6
            if let Some(payload) = GenesisPayload::from_bytes(&buf[6..]) {
                // Recupera os pesos biológicos
                if let Some((fast_weights, slow, generation, fitness)) = decompress_organism(&payload) {
                    let slow_weights = if slow.is_empty() { None } else { Some(slow) };
                    
                    // Recupera metadados do Universo
                    if let Some(meta_chunk) = payload.find_chunk(TAG_META_UNIVERSE) {
                        let meta_buf = &meta_chunk.data;
                        if meta_buf.len() >= 16 {
                            let seed = u32::from_le_bytes(meta_buf[0..4].try_into().unwrap());
                            let track_width = f32::from_le_bytes(meta_buf[4..8].try_into().unwrap());
                            let fast_scale = f32::from_le_bytes(meta_buf[8..12].try_into().unwrap());
                            let slow_scale = f32::from_le_bytes(meta_buf[12..16].try_into().unwrap());
                            
                            // Morpho (offset 16..32 se existir, senao default)
                            let morpho = if meta_buf.len() >= 32 {
                                MorphoGenes::from_le_bytes(&meta_buf[16..32]).unwrap_or_default()
                            } else {
                                MorphoGenes::default()
                            };

                            // Hypernet (offset 32..292 se existir, senao default)
                            let hypernet = if meta_buf.len() >= 32 + HYPERNET_BYTES {
                                HypernetSeed::from_le_bytes(&meta_buf[32..32 + HYPERNET_BYTES]).unwrap_or_default()
                            } else {
                                HypernetSeed::default()
                            };

                            // KAN knots (offset 292..436 se existir; default tanh ramp caso contrário)
                            let knots_start = 32 + HYPERNET_BYTES;
                            let kan_knots = if meta_buf.len() >= knots_start + KAN_KNOTS_BYTES {
                                kan_knots_from_bytes(&meta_buf[knots_start..knots_start + KAN_KNOTS_BYTES])
                                    .unwrap_or_else(default_kan_knots)
                            } else {
                                default_kan_knots()
                            };

                            // Trinity chunk (opcional, ausente em DNAs antigos).
                            let trinity = payload.find_chunk(TAG_TRINITY)
                                .and_then(|c| TrinityWeights::from_bytes(&c.data));
                            // CEQU chunk (cortex equation, opcional).
                            let cortex_dct = payload.find_chunk(TAG_CORTEX_EQ)
                                .and_then(|c| dct_coeffs_from_bytes(&c.data));
                            // CRAW chunk (cortex raw i8) — bypass compress_organism ternary.
                            let cortex_raw: Option<Vec<i8>> = payload.find_chunk(TAG_CORTEX_RAW)
                                .map(|c| c.data.iter().map(|&b| b as i8).collect());
                            // CL2W chunk (V7): cortex layer 2 weights raw i8 (64×32=2048).
                            let cortex_l2_raw: Option<Vec<i8>> = payload.find_chunk(TAG_CORTEX_L2)
                                .map(|c| c.data.iter().map(|&b| b as i8).collect());
                            // SLWF chunk: slow_brain raw f32 (lossless, bypass ternary).
                            let slow_f32: Option<Vec<f32>> = payload.find_chunk(TAG_SLOW_F32)
                                .and_then(|c| {
                                    if c.data.len() % 4 != 0 { return None; }
                                    Some(c.data.chunks_exact(4)
                                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                        .collect())
                                });
                            return Some(DnaPayload {
                car_model_idx: 0,
                                fast_weights, fast_scale,
                                slow_weights, slow_scale,
                                generation, seed, track_width, fitness, version, morpho, hypernet, kan_knots,
                                trinity,
                                cortex_dct,
                                cortex_raw,
                                cortex_l2_raw,
                                cortex_l3_raw: None,
                                slow_f32,
                            });
                        }
                    }
                }
            }
        }
    }

    // --- v3 with magic ---
    if buf.len() >= DNA_V3_TOTAL && &buf[0..4] == DNA_MAGIC {
        let version = u16::from_le_bytes(buf[4..6].try_into().ok()?);
        if version != 4 {
            let generation = u32::from_le_bytes(buf[6..10].try_into().ok()?);
            let seed = u32::from_le_bytes(buf[10..14].try_into().ok()?);
            let track_width = f32::from_le_bytes(buf[14..18].try_into().ok()?);
            let fitness = f32::from_le_bytes(buf[18..22].try_into().ok()?);
            let fast_scale_raw = f32::from_le_bytes(buf[22..26].try_into().ok()?);
            let fast_end = 26 + DNA_FAST_SIZE_LEGACY;
            let fast_weights: Vec<i8> = buf[26..fast_end].iter().map(|&b| b as i8).collect();
            let slow_scale_raw = f32::from_le_bytes(buf[fast_end..fast_end + 4].try_into().ok()?);
            let slow_start = fast_end + 4;

        let slow_end = slow_start + DNA_SLOW_SIZE_LEGACY;
        if buf.len() < slow_end { return None; }
        let slow_weights: Vec<i8> = buf[slow_start..slow_end].iter().map(|&b| b as i8).collect();
        // SANITIZAÇÃO: DNAs antigos podem ter scales astronômicos (bug myelinate
        // sem cap pré-fix). Clamp em [1e-4, 10] — valores fora disso indicam
        // corrupção runtime, não pesos válidos. Força recalibração na próxima
        // quantize() via abs_mean dos pesos ternários.
        let fast_scale = if fast_scale_raw.is_finite() { fast_scale_raw.clamp(1e-4, 10.0) } else { 1.0 };
        let slow_scale = if slow_scale_raw.is_finite() { slow_scale_raw.clamp(1e-4, 10.0) } else { 1.0 };
        // --- Tier 6.3 LITE: tail opcional de morpho (backward-compat c/ v3 antigo) ---
        let morpho = if buf.len() >= slow_end + MORPHO_BYTES {
            MorphoGenes::from_le_bytes(&buf[slow_end..slow_end + MORPHO_BYTES])
                .unwrap_or_default()
        } else {
            MorphoGenes::default()
        };
        // --- Tier 6.4 LITE: tail opcional de hypernet (backward-compat) ---
        let hypernet_start = slow_end + MORPHO_BYTES;
        let hypernet = if buf.len() >= hypernet_start + HYPERNET_BYTES {
            HypernetSeed::from_le_bytes(&buf[hypernet_start..hypernet_start + HYPERNET_BYTES])
                .unwrap_or_default()
        } else {
            HypernetSeed::default()
        };
        let knots_start = hypernet_start + HYPERNET_BYTES;
        let kan_knots = if buf.len() >= knots_start + KAN_KNOTS_BYTES {
            kan_knots_from_bytes(&buf[knots_start..knots_start + KAN_KNOTS_BYTES])
                .unwrap_or_else(default_kan_knots)
        } else {
            default_kan_knots()
        };
        return Some(DnaPayload {
                car_model_idx: 0,
            fast_weights, fast_scale,
            slow_weights: Some(slow_weights), slow_scale,
            generation, seed, track_width, fitness, version, morpho, hypernet, kan_knots,
            trinity: None,
            cortex_dct: None,
            cortex_raw: None,
            cortex_l2_raw: None,
            cortex_l3_raw: None,
            slow_f32: None,
        });
        }
    }

    // --- Legacy v1/v2 (no magic) ---
    if buf.len() >= 4 + DNA_FAST_SIZE_LEGACY {
        let fast_scale = f32::from_le_bytes(buf[0..4].try_into().ok()?);
        let fast_end = 4 + DNA_FAST_SIZE_LEGACY;
        let fast_weights: Vec<i8> = buf[4..fast_end].iter().map(|&b| b as i8).collect();
        let (slow_weights, slow_scale, version) = if buf.len() >= fast_end + 4 + DNA_SLOW_SIZE_LEGACY {
            let ss = f32::from_le_bytes(buf[fast_end..fast_end + 4].try_into().ok()?);
            let sw: Vec<i8> = buf[fast_end + 4..fast_end + 4 + DNA_SLOW_SIZE_LEGACY].iter().map(|&b| b as i8).collect();
            (Some(sw), ss, 2u16)
        } else {
            (None, 1.0, 1u16)
        };
        return Some(DnaPayload {
                car_model_idx: 0,
            fast_weights, fast_scale,
            slow_weights, slow_scale,
            generation: 0, seed: 0, track_width: 0.0, fitness: 0.0, version,
            morpho: MorphoGenes::default(),
            hypernet: HypernetSeed::default(),
            kan_knots: default_kan_knots(),
            trinity: None,
            cortex_dct: None,
            cortex_raw: None,
            cortex_l2_raw: None,
            cortex_l3_raw: None,
            slow_f32: None,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn dct_roundtrip_smooth_signal() {
        // Smooth f(i) = sin(2πi/N) + 0.5*cos(4πi/N) — should reconstruct accurately.
        let n = CORTEX_FULL_SIZE;
        let original: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / n as f32;
            (2.0 * std::f32::consts::PI * t).sin() + 0.5 * (4.0 * std::f32::consts::PI * t).cos()
        }).collect();
        let coeffs = cortex_to_dct(&original);
        let recovered = cortex_from_dct(&coeffs);
        let max_err = original.iter().zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 0.01, "smooth round-trip max err {} too high", max_err);
    }

    #[test]
    fn dct_roundtrip_random_signal_lossy_but_bounded() {
        // Random weights — 32× compression IS lossy, but RMS should stay reasonable.
        let n = CORTEX_FULL_SIZE;
        let mut s = 12345_u64;
        let original: Vec<f32> = (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }).collect();
        let coeffs = cortex_to_dct(&original);
        let recovered = cortex_from_dct(&coeffs);
        let mse = original.iter().zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / n as f32;
        // Random signal lossy 32× → MSE ~ var × (1 - K/N) = 0.33 × 0.97 ≈ 0.32
        assert!(mse < 0.4, "random-signal MSE {} unreasonable", mse);
    }

    fn sample_weights(offset: i8) -> (Vec<i8>, Vec<i8>) {
        let fast: Vec<i8> = (0..DNA_FAST_SIZE).map(|i| (((i as i32 + offset as i32) % 3) - 1) as i8).collect();
        let slow: Vec<i8> = (0..DNA_SLOW_SIZE).map(|i| (((i as i32 + offset as i32) % 3) - 1) as i8).collect();
        (fast, slow)
    }

    #[test]
    fn v3_round_trip_preserves_everything() {
        let (fast, slow) = sample_weights(0);
        let path = std::env::temp_dir().join("iara_test_v3_genesis.dna");
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
        let path = std::env::temp_dir().join("iara_test_v2_genesis.dna");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0.5_f32.to_le_bytes()).unwrap();
        f.write_all(&fast.iter().map(|&w| w as u8).collect::<Vec<u8>>()).unwrap();
        f.write_all(&0.25_f32.to_le_bytes()).unwrap();
        f.write_all(&slow.iter().map(|&w| w as u8).collect::<Vec<u8>>()).unwrap();
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
        // Verifica consistencia v3 legacy (usando os layouts legacy):
        assert_eq!(DNA_V3_TOTAL, 4 + 2 + 4 + 4 + 4 + 4 + 4 + DNA_FAST_SIZE_LEGACY + 4 + DNA_SLOW_SIZE_LEGACY);
        assert_eq!(DNA_V31_TOTAL, DNA_V3_TOTAL + MORPHO_BYTES);
    }

    #[test]
    fn morpho_round_trip_and_file_size() {
        let (fast, slow) = sample_weights(2);
        let path = std::env::temp_dir().join("iara_test_v31_v4_morpho.dna");
        let path_s = path.to_str().unwrap();
        let morpho = MorphoGenes { mass: 1234.5, wheelbase: 2.1, downforce: 1.7, grip_mult: 0.9 };
        save_champion_dna_with_morpho(path_s, &fast, 0.1, &slow, 0.2, 1, 2, 10.0, 50.0, morpho).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        // Em V4 (Generative Compression) file size vai variar baseado na entropia da sequência.
        assert!(meta.len() > 0);
        let loaded = load_champion_dna(path_s).expect("v4 must load");
        assert!((loaded.morpho.mass - 1234.5).abs() < 1e-1);
        assert!((loaded.morpho.wheelbase - 2.1).abs() < 1e-5);
        assert!((loaded.morpho.downforce - 1.7).abs() < 1e-5);
        assert!((loaded.morpho.grip_mult - 0.9).abs() < 1e-5);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn morpho_sanitize_clamps_out_of_range() {
        let wild = MorphoGenes { mass: 9000.0, wheelbase: 0.1, downforce: -5.0, grip_mult: 99.0 };
        let s = wild.sanitized();
        assert!(s.mass <= MORPHO_MASS_MAX && s.mass >= MORPHO_MASS_MIN);
        assert!(s.wheelbase <= MORPHO_WHEELBASE_MAX && s.wheelbase >= MORPHO_WHEELBASE_MIN);
        assert!(s.downforce <= MORPHO_DOWNFORCE_MAX && s.downforce >= MORPHO_DOWNFORCE_MIN);
        assert!(s.grip_mult <= MORPHO_GRIP_MAX && s.grip_mult >= MORPHO_GRIP_MIN);
    }

    #[test]
    fn legacy_v3_without_morpho_loads_with_default() {
        // Simula um arquivo v3 salvo pre-tier-6.3 (8478 bytes, sem morpho tail).
        let (fast, slow) = sample_weights(3);
        let path = std::env::temp_dir().join("iara_test_v3_legacy_nomorpho.dna");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(DNA_MAGIC).unwrap();
        f.write_all(&3_u16.to_le_bytes()).unwrap(); // Legacy V3 tem version=3
        f.write_all(&7_u32.to_le_bytes()).unwrap();
        f.write_all(&123_u32.to_le_bytes()).unwrap();
        f.write_all(&12.0_f32.to_le_bytes()).unwrap();
        f.write_all(&42.0_f32.to_le_bytes()).unwrap();
        f.write_all(&0.3_f32.to_le_bytes()).unwrap();
        f.write_all(&fast.iter().map(|&w| w as u8).collect::<Vec<u8>>()).unwrap();
        f.write_all(&0.4_f32.to_le_bytes()).unwrap();
        f.write_all(&slow.iter().map(|&w| w as u8).collect::<Vec<u8>>()).unwrap();
        drop(f);
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len() as usize, DNA_V3_TOTAL);
        let loaded = load_champion_dna(path.to_str().unwrap()).expect("legacy v3 must load");
        let d = MorphoGenes::default();
        assert!((loaded.morpho.mass - d.mass).abs() < 1e-3);
        assert!((loaded.morpho.wheelbase - d.wheelbase).abs() < 1e-5);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn kan_knots_round_trip_v4() {
        let (fast, slow) = sample_weights(9);
        let path = std::env::temp_dir().join("iara_test_kan_knots.dna");
        let path_s = path.to_str().unwrap();
        let mut knots = default_kan_knots();
        // Alvo distinguível do default pra confirmar que a escrita preserva valores.
        for o in 0..3 {
            for k in 0..crate::kan::KAN_OUT_KNOTS {
                knots[o][k] = ((o as f32) * 0.3) + ((k as f32) * 0.05) - 0.5;
            }
        }
        save_champion_dna_full(
            path_s, &fast, 0.5, &slow, 0.25, 10, 20, 14.0, 777.0,
            MorphoGenes::default(), HypernetSeed::default(), knots,
        ).unwrap();
        let loaded = load_champion_dna(path_s).expect("v4 kan round trip");
        for o in 0..3 {
            for k in 0..crate::kan::KAN_OUT_KNOTS {
                assert!((loaded.kan_knots[o][k] - knots[o][k]).abs() < 1e-5,
                    "knot mismatch at [{}][{}]: {} vs {}", o, k, loaded.kan_knots[o][k], knots[o][k]);
            }
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn legacy_v4_without_knots_loads_with_default() {
        // V4 salvo com assinatura antiga (sem knots tail) → default_kan_knots().
        let (fast, slow) = sample_weights(5);
        let path = std::env::temp_dir().join("iara_test_v4_no_knots.dna");
        let path_s = path.to_str().unwrap();
        save_champion_dna_with_morpho(
            path_s, &fast, 0.1, &slow, 0.2, 1, 2, 10.0, 50.0, MorphoGenes::default(),
        ).unwrap();
        let loaded = load_champion_dna(path_s).expect("must load");
        let def = default_kan_knots();
        for o in 0..3 {
            for k in 0..crate::kan::KAN_OUT_KNOTS {
                assert!((loaded.kan_knots[o][k] - def[o][k]).abs() < 1e-5);
            }
        }
        let _ = std::fs::remove_file(path);
    }
}
