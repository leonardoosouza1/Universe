//! MOVED out of the gpu_evolve binary into the library (2026-08-14, P1.9): genoma_view
//! kept a DUPLICATE parser that broke in silence when the P1.6/P1.7 blocks entered the
//! format (it read neuron_activation believing it was τ — reporting 'spread 0.0×' for a
//! champion whose real spread was 3.2×). The canonical parser lives HERE; binaries import
//! it. On-disk genome persistence: `save_genome` / `load_genome`.
//!
//! # Format v2 (with a header) — E3 (REFACTOR_PLAN.md §2.3, item 3)
//!
//! ```text
//! magic:   4 bytes  = b"UGEN"
//! version: 4 bytes  = u32 LE = 2
//! L1:      N_L1_WEIGHTS  × f32 LE
//! L2:      N_L2_WEIGHTS  × f32 LE
//! KAN:     3*KAN_OUT_KNOTS × f32 LE
//! slow:    N_SLOW_WEIGHTS × f32 LE   (optional — read only if the file is large enough)
//! tau:     N_CFC_CHANNELS × f32 LE   (optional — same rule)
//! sensors: N_SENSOR_GENES × f32 LE   (optional — same rule; SENSOR PHASE, step 1)
//! neuron_alpha:       (L1+L2) × f32 LE  (optional — same rule; P1.7, mechanism 1)
//! neuron_period_l1:   L1      × f32 LE  (optional — same rule; P1.7, mechanism 2)
//! neuron_threshold_l1: L1     × f32 LE  (optional — same rule; P1.7, mechanism 2)
//! neuron_activation:  (L1+L2) × f32 LE  (optional — same rule; P1.7, mechanism 3)
//! ```
//!
//! # Compatibility with the legacy format (NO header)
//!
//! Before this version the file started straight at the L1 bytes — no magic, no version.
//! `load_genome` *sniffs* the first 4 bytes: if they are exactly `b"UGEN"`, it reads the
//! version and skips the 8-byte header; otherwise it assumes the legacy format and reads
//! the blocks from byte 0, exactly as before.
//!
//! That preserves the original guarantee: a legacy `.bin` written before the slow brain
//! and the τ bank existed still loads (falling back to the default documented in
//! `load_genome`), and now a v2 `.bin` is recognised unambiguously too — the sniff can
//! never confuse the two, because 4 bytes of random f32 weight colliding with exactly
//! `b"UGEN"` has probability ~2⁻³² per file.
//!
//! **DO NOT reorder `Genome`'s fields or the write order below** — the disk has no field
//! names, only positions. Reordering corrupts existing `.bin` files silently (rule R5 of
//! REFACTOR_PLAN.md). Every new field in `Genome` requires updating `save_genome` AND
//! `load_genome` together — the tests in `tests.rs` catch the omission.

use crate::gpu::evolution::Genome;
use crate::kan::KAN_OUT_KNOTS;
use crate::evolution::dna_genesis::default_kan_knots;

/// Magic of the v2 header — 4 ASCII bytes, "Universe GENome".
const MAGIC: &[u8; 4] = b"UGEN";
/// Current format version. Bump it here WHENEVER the block layout changes (new field, new
/// order) — `load_genome` decides what to read from it.
const FORMAT_VERSION: u32 = 2;

/// Saves the best genome: v2 header + L1 | L2 | KAN | slow | τ bank, all f32 little-endian.
///
/// The slow brain was historically missing. Saving only L1/L2/KAN meant the champion lost
/// the slow brain's 1091 weights when reloaded through `--carregar`: it came back with the
/// default initialisation, identical for everyone. Generations of selection over modulation
/// were silently discarded on every resume. (`--carregar` is the flag name from when the
/// bug happened; the resume flag today is `--retomar`, parsed at
/// `src/bin/gpu_evolve/main.rs:741`.)
pub fn save_genome(genome: &Genome, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&FORMAT_VERSION.to_le_bytes())?;
    for w in &genome.l1 { f.write_all(&w.to_le_bytes())?; }
    for w in &genome.l2 { f.write_all(&w.to_le_bytes())?; }
    for row in &genome.kan_knots {
        for k in row { f.write_all(&k.to_le_bytes())?; }
    }
    for w in &genome.slow { f.write_all(&w.to_le_bytes())?; }
    for d in &genome.tau  { f.write_all(&d.to_le_bytes())?; }
    for s in &genome.sensors { f.write_all(&s.to_le_bytes())?; }
    // P1.7 (docs/PLANO_EVOLUCAO.md) — asynchronous brain: memory, period/threshold,
    // heterogeneous activation. Same rule as the blocks above: a new field here WITHOUT
    // updating `load_genome` alongside it is gap 7 of the project's CLAUDE.md (we lost
    // 1091 weights in silence once by forgetting exactly this) — `tests.rs` covers it.
    for a in &genome.neuron_alpha { f.write_all(&a.to_le_bytes())?; }
    for p in &genome.neuron_period_l1 { f.write_all(&p.to_le_bytes())?; }
    for t in &genome.neuron_threshold_l1 { f.write_all(&t.to_le_bytes())?; }
    for a in &genome.neuron_activation { f.write_all(&a.to_le_bytes())?; }
    // M4: the maestro's gates (16 f32) — optional block, always last.
    for a in &genome.gates_w { f.write_all(&a.to_le_bytes())?; }
    Ok(())
}

/// Tries to load a genome from a file written by `save_genome` — v2 (with the `UGEN`
/// header) OR legacy (no header, the L1 bytes straight at the start of the file).
pub fn load_genome(path: &str) -> Option<Genome> {
    let bytes = std::fs::read(path).ok()?;

    // Sniff the first 4 bytes: separates v2 from legacy with no practical ambiguity.
    let base_off = if bytes.len() >= 8 && &bytes[0..4] == MAGIC {
        let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        // Only v2 exists so far; an unknown (higher) version still attempts to read with
        // the current layout — a file of a LOWER version should not carry the magic.
        debug_assert!(version <= FORMAT_VERSION, "genome saved by a future version ({version})");
        8usize
    } else {
        0usize
    };

    let n_l1 = crate::gpu::evolution::N_L1_WEIGHTS;
    let n_l2 = crate::gpu::evolution::N_L2_WEIGHTS;
    let n_kan = 3 * KAN_OUT_KNOTS;
    let expected = base_off + (n_l1 + n_l2 + n_kan) * 4;
    if bytes.len() < expected { return None; }

    // Reading by explicit offset. The version with a closure mutating `off` does not pass
    // the borrow checker as soon as someone needs to read `off` from outside it — and
    // needing to is unavoidable here, because the final blocks are optional and the
    // decision to read them depends on comparing `off` against the file size.
    fn read_at(bytes: &[u8], off: usize, n: usize) -> Vec<f32> {
        bytes[off..off + n * 4].chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    let mut off = base_off;
    let l1 = read_at(&bytes, off, n_l1);   off += n_l1 * 4;
    let l2 = read_at(&bytes, off, n_l2);   off += n_l2 * 4;
    let kan_flat = read_at(&bytes, off, n_kan); off += n_kan * 4;
    let mut kan_knots = default_kan_knots();
    for o in 0..3 {
        for k in 0..KAN_OUT_KNOTS {
            kan_knots[o][k] = kan_flat[o * KAN_OUT_KNOTS + k];
        }
    }
    // Optional blocks, read only if the file contains them. A file written before these
    // fields existed falls back to the default — which is the truth, not an invention:
    // that genome really did not carry the data.
    let n_slow = crate::engine::SlowBrain::new().all_weights().len();
    let n_tau  = crate::gpu::evolution::N_CFC_CHANNELS;

    let slow = if bytes.len() >= off + n_slow * 4 {
        let v = read_at(&bytes, off, n_slow); off += n_slow * 4; v
    } else {
        crate::engine::SlowBrain::new().all_weights()
    };
    let tau = if bytes.len() >= off + n_tau * 4 {
        let v = read_at(&bytes, off, n_tau); off += n_tau * 4; v
    } else {
        vec![crate::cfc::DECAY_DEFAULT; n_tau]
    };
    // SENSOR PHASE (step 1) — same optional pattern as slow/tau: a file written before
    // this field existed falls back to the default, which is the truth (that genome really
    // did run with today's hardcoded values, not with an invented one).
    let n_sensors = crate::gpu::evolution::N_SENSOR_GENES;
    let sensors = if bytes.len() >= off + n_sensors * 4 {
        let v = read_at(&bytes, off, n_sensors); off += n_sensors * 4; v
    } else {
        crate::gpu::evolution::SENSOR_DEFAULTS.to_vec()
    };

    // P1.7 — same optional pattern as slow/tau/sensors: a file written before these fields
    // existed falls back to NEUTRAL (identical to today's brain — see `Genome::random`),
    // which is the truth, not an invention: that genome really did run without
    // memory/period/threshold/heterogeneous activation.
    let n_l1 = crate::gpu::evolution::N_L1_NEURONS;
    let n_l2 = crate::gpu::evolution::N_L2_NEURONS;
    let neuron_alpha = if bytes.len() >= off + (n_l1 + n_l2) * 4 {
        let v = read_at(&bytes, off, n_l1 + n_l2); off += (n_l1 + n_l2) * 4; v
    } else {
        vec![0.0; n_l1 + n_l2]
    };
    let neuron_period_l1 = if bytes.len() >= off + n_l1 * 4 {
        let v = read_at(&bytes, off, n_l1); off += n_l1 * 4; v
    } else {
        vec![1.0; n_l1]
    };
    let neuron_threshold_l1 = if bytes.len() >= off + n_l1 * 4 {
        let v = read_at(&bytes, off, n_l1); off += n_l1 * 4; v
    } else {
        vec![0.0; n_l1]
    };
    let neuron_activation = if bytes.len() >= off + (n_l1 + n_l2) * 4 {
        let v = read_at(&bytes, off, n_l1 + n_l2); off += (n_l1 + n_l2) * 4; v
    } else {
        vec![0.0; n_l1 + n_l2]
    };
    // M4: the maestro's gates (optional; an old genome → zeros = neutral gates of 1.0).
    let gates_w = if bytes.len() >= off + 16 * 4 {
        read_at(&bytes, off, 16)
    } else {
        vec![0.0; 16]
    };

    Some(Genome {
        l1, l2, kan_knots, slow, tau, sensors,
        neuron_alpha, neuron_period_l1, neuron_threshold_l1, neuron_activation,
        gates_w,
    })
}

/// Seed of being `i`'s initial genome under the population seed `pop_seed`.
///
/// This used to exist in THREE separate branches (checkpoint failed / no checkpoint / from
/// scratch) and only one of them was seeded, without checking which one actually runs. The
/// result: a whole battery of 6 runs with "three seeds" that were the SAME population —
/// `gen 0` identical down to the wall-clock time in all three, and a paired effect of 25σ
/// that was one sample repeated three times.
///
/// With a single function, forgetting to seed a branch is impossible.
pub fn population_seed_for(pop_seed: u64, index: usize) -> u64 {
    // A large prime multiplier separates the seed families: populations from different
    // `pop_seed` values share no initial genome.
    pop_seed.wrapping_mul(1_000_003).wrapping_add(index as u64)
}
