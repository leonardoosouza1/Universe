//! MOVED out of the gpu_evolve binary into the library (2026-08-14, P1.9): genoma_view
//! kept a DUPLICATE parser that broke in silence when the P1.6/P1.7 blocks entered the
//! format (it read neuron_activation believing it was τ — reporting 'spread 0.0×' for a
//! champion whose real spread was 3.2×). The canonical parser lives HERE; binaries import
//! it. On-disk genome persistence: `save_genome` / `load_genome`.
//!
//! # Format v5 (organism-taxi-v1 hereditary controls)
//! `UGEN | version=5:u32 | payload_bytes:u32 | body[2] | organs[2] | life[16] | organism[16] | v2_payload`.
//! Organism controls are finite in [-1,1]; acquired state belongs in a snapshot.
//! A neutral organism block preserves v2/v3/v4 encoding.
//!
//! # Format v4 (body, organs and lifetime DNA)
//! `UGEN | version=4:u32 | payload_bytes:u32 | body[2] | organs[2] | life[16] | v2_payload`.
//! Neutral new extensions retain v2/v3 encoding exactly. Acquired plastic weights and
//! eligibility traces are never serialized into hereditary DNA.
//!
//! # Format v3 (experimental body genes)
//! `UGEN | version=3:u32 | payload_bytes:u32 | body:[f32;2] | v2_payload`.
//! All numbers are little-endian. Body values are finite and in [-1,1]. The payload
//! retains the exact v2 field order including its variable-length Fly tail. Neutral
//! bodies are still saved as v2. Unknown versions fail closed in release builds too.
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
//! gates_w:            16      × f32 LE  (optional — same rule; M4 conductor gates)
//! hip:                29      × f32 LE  (optional — same rule; C1 sector hippocampus)
//! fly:          200+n_types    × f32 LE  (optional; FLY MAP selection, ALWAYS LAST —
//!                                        length = whatever is left in the file)
//! ```
//!
//! # The one block whose length is not a constant
//!
//! Every block above has a length fixed at compile time, so `load_genome` knows where the
//! next one starts. `fly` does not: it is `200 + n_types` and `n_types` comes from the map
//! file the run loaded, which a reader of the `.bin` may not have. That is exactly why it
//! is LAST: whatever remains after `hip` is the fly block, and nothing depends on
//! knowing its length in advance. A block written against another map therefore loads at
//! its own length and is treated as inert by `fly_map::pack_selection`, instead of being
//! read at the wrong stride.
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

use crate::evolution::dna_genesis::default_kan_knots;
use crate::gpu::body::BodyGenes;
use crate::gpu::evolution::Genome;
use crate::gpu::lifetime::{LifeGenes, N_LIFE_GENES};
use crate::gpu::organs::OrganGenes;
use crate::kan::KAN_OUT_KNOTS;

/// Magic of the v2 header — 4 ASCII bytes, "Universe GENome".
const MAGIC: &[u8; 4] = b"UGEN";
/// Current format version. Bump it here WHENEVER the block layout changes (new field, new
/// order) — `load_genome` decides what to read from it.
const FORMAT_VERSION: u32 = 5;

/// Saves hereditary DNA using the earliest compatible v2/v3/v4/v5 envelope.
///
/// The slow brain was historically missing. Saving only L1/L2/KAN meant the champion lost
/// the slow brain's 1091 weights when reloaded through `--carregar`: it came back with the
/// default initialisation, identical for everyone. Generations of selection over modulation
/// were silently discarded on every resume. (`--carregar` is the flag name from when the
/// bug happened; the resume flag today is `--resume`, parsed at
/// `src/bin/gpu_evolve/main.rs:741`.)
pub fn save_genome(genome: &Genome, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    if !genome.body.is_valid()
        || !genome.organs.is_valid()
        || !genome.lifetime.is_valid()
        || !super::body::organism_genes_valid(&genome.organism)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid body, organ, lifetime or organism genes",
        ));
    }
    // Neutral genomes keep their exact historical v2 bytes. A v3 envelope adds
    // payload length and body genes BEFORE the unchanged positional payload,
    // so the variable-length Fly tail is never mistaken for body genes.
    let version = if genome.organism != [0.0; 16] {
        FORMAT_VERSION
    } else if genome.organs != OrganGenes::default() || genome.lifetime != LifeGenes::default() {
        4
    } else if genome.body != BodyGenes::default() {
        3
    } else {
        2
    };
    if version == 5 && !valid_v5_payload(genome) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "v5 requires a complete finite neural payload",
        ));
    }
    let payload_floats = genome.l1.len()
        + genome.l2.len()
        + 3 * KAN_OUT_KNOTS
        + genome.slow.len()
        + genome.tau.len()
        + genome.sensors.len()
        + genome.neuron_alpha.len()
        + genome.neuron_period_l1.len()
        + genome.neuron_threshold_l1.len()
        + genome.neuron_activation.len()
        + genome.gates_w.len()
        + genome.hip.len()
        + genome.fly.len();
    let payload_bytes = u32::try_from(payload_floats * 4).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "genome payload too large")
    })?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&version.to_le_bytes())?;
    if version >= 3 {
        f.write_all(&payload_bytes.to_le_bytes())?;
        for gene in genome.body.0 {
            f.write_all(&gene.to_le_bytes())?;
        }
    }
    if version >= 4 {
        for gene in genome.organs.0.iter().chain(genome.lifetime.0.iter()) {
            f.write_all(&gene.to_le_bytes())?;
        }
    }
    if version >= 5 {
        for gene in genome.organism {
            f.write_all(&gene.to_le_bytes())?;
        }
    }
    for w in &genome.l1 {
        f.write_all(&w.to_le_bytes())?;
    }
    for w in &genome.l2 {
        f.write_all(&w.to_le_bytes())?;
    }
    for row in &genome.kan_knots {
        for k in row {
            f.write_all(&k.to_le_bytes())?;
        }
    }
    for w in &genome.slow {
        f.write_all(&w.to_le_bytes())?;
    }
    for d in &genome.tau {
        f.write_all(&d.to_le_bytes())?;
    }
    for s in &genome.sensors {
        f.write_all(&s.to_le_bytes())?;
    }
    // P1.7 (docs/PLANO_EVOLUCAO.md) — asynchronous brain: memory, period/threshold,
    // heterogeneous activation. Same rule as the blocks above: a new field here WITHOUT
    // updating `load_genome` alongside it is gap 7 of the project's CLAUDE.md (we lost
    // 1091 weights in silence once by forgetting exactly this) — `tests.rs` covers it.
    for a in &genome.neuron_alpha {
        f.write_all(&a.to_le_bytes())?;
    }
    for p in &genome.neuron_period_l1 {
        f.write_all(&p.to_le_bytes())?;
    }
    for t in &genome.neuron_threshold_l1 {
        f.write_all(&t.to_le_bytes())?;
    }
    for a in &genome.neuron_activation {
        f.write_all(&a.to_le_bytes())?;
    }
    // M4: the maestro's gates (16 f32) — optional block.
    for a in &genome.gates_w {
        f.write_all(&a.to_le_bytes())?;
    }
    // C1 (docs/PRD-CEREBRO-NEUROMOD.md §C1): sector hippocampus, 29 f32 — optional block,
    // ALWAYS LAST. A file written before C1 existed simply ends here, and `load_genome`
    // falls back to all-zero, which is the truth and not an invention: that genome really
    // did run without a hippocampus, and all-zero is exactly the "no hippocampus"
    // behaviour (bias `[0, 0, 0]`, see `gpu::hippocampus`).
    for a in &genome.hip {
        f.write_all(&a.to_le_bytes())?;
    }
    // FLY MAP (docs/FLY-MAP-BRAIN.md §3): the selection on top of the connectome — optional
    // block, ALWAYS LAST, and the only one whose length is not a compile-time constant (see
    // the module header). Empty when the run had no `--fly-map`, in which case the file ends
    // exactly where it used to and `load_genome` falls back to an empty block.
    for a in &genome.fly {
        f.write_all(&a.to_le_bytes())?;
    }
    Ok(())
}

// A v5 envelope cannot silently shift fields through malformed vector lengths.
fn valid_v5_payload(genome: &Genome) -> bool {
    use super::evolution::*;
    let fields: &[(&[f32], usize)] = &[
        (&genome.l1, N_L1_WEIGHTS),
        (&genome.l2, N_L2_WEIGHTS),
        (
            &genome.slow,
            crate::engine::SlowBrain::new().all_weights().len(),
        ),
        (&genome.tau, N_CFC_CHANNELS),
        (&genome.sensors, N_SENSOR_GENES),
        (&genome.neuron_alpha, N_L1_NEURONS + N_L2_NEURONS),
        (&genome.neuron_period_l1, N_L1_NEURONS),
        (&genome.neuron_threshold_l1, N_L1_NEURONS),
        (&genome.neuron_activation, N_L1_NEURONS + N_L2_NEURONS),
        (&genome.gates_w, 16),
        (&genome.hip, super::hippocampus::N_HIP_GENES),
    ];
    fields
        .iter()
        .all(|(values, expected)| values.len() == *expected && values.iter().all(|v| v.is_finite()))
        && genome
            .kan_knots
            .iter()
            .flatten()
            .chain(genome.fly.iter())
            .all(|v| v.is_finite())
}

/// Tries to load a genome from a file written by `save_genome` — v2/v3 (with the `UGEN`
/// header) OR legacy (no header, the L1 bytes straight at the start of the file).
pub fn load_genome(path: &str) -> Option<Genome> {
    let bytes = std::fs::read(path).ok()?;

    // Sniff the first 4 bytes: separates v2 from legacy with no practical ambiguity.
    let (base_off, body, organs, lifetime, organism) = if bytes.starts_with(MAGIC) {
        if bytes.len() < 8 {
            return None;
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        match version {
            2 => (
                8usize,
                BodyGenes::default(),
                OrganGenes::default(),
                LifeGenes::default(),
                [0.0; 16],
            ),
            3 | 4 | 5 => {
                let header_bytes =
                    20 + if version >= 4 {
                        (2 + N_LIFE_GENES) * 4
                    } else {
                        0
                    } + if version >= 5 { 16 * 4 } else { 0 };
                if bytes.len() < header_bytes {
                    return None;
                }
                let length = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
                if length % 4 != 0 || length.checked_add(header_bytes)? != bytes.len() {
                    return None;
                }
                let body = BodyGenes([
                    f32::from_le_bytes(bytes[12..16].try_into().ok()?),
                    f32::from_le_bytes(bytes[16..20].try_into().ok()?),
                ]);
                if !body.is_valid() {
                    return None;
                }
                let mut organs = OrganGenes::default();
                let mut lifetime = LifeGenes::default();
                if version >= 4 {
                    for (i, gene) in organs.0.iter_mut().chain(lifetime.0.iter_mut()).enumerate() {
                        let offset = 20 + i * 4;
                        *gene = f32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
                    }
                    if !organs.is_valid() || !lifetime.is_valid() {
                        return None;
                    }
                }
                let mut organism = [0.0; 16];
                if version >= 5 {
                    for (i, gene) in organism.iter_mut().enumerate() {
                        let offset = 20 + (2 + N_LIFE_GENES + i) * 4;
                        *gene = f32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
                    }
                    if !super::body::organism_genes_valid(&organism) {
                        return None;
                    }
                    if bytes[header_bytes..]
                        .chunks_exact(4)
                        .any(|chunk| !f32::from_le_bytes(chunk.try_into().unwrap()).is_finite())
                    {
                        return None;
                    }
                }
                (header_bytes, body, organs, lifetime, organism)
            }
            _ => return None,
        }
    } else {
        (
            0usize,
            BodyGenes::default(),
            OrganGenes::default(),
            LifeGenes::default(),
            [0.0; 16],
        )
    };

    let n_l1 = crate::gpu::evolution::N_L1_WEIGHTS;
    let n_l2 = crate::gpu::evolution::N_L2_WEIGHTS;
    let n_kan = 3 * KAN_OUT_KNOTS;
    let expected = base_off + (n_l1 + n_l2 + n_kan) * 4;
    if bytes.len() < expected {
        return None;
    }

    // Reading by explicit offset. The version with a closure mutating `off` does not pass
    // the borrow checker as soon as someone needs to read `off` from outside it — and
    // needing to is unavoidable here, because the final blocks are optional and the
    // decision to read them depends on comparing `off` against the file size.
    fn read_at(bytes: &[u8], off: usize, n: usize) -> Vec<f32> {
        bytes[off..off + n * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    let mut off = base_off;
    let l1 = read_at(&bytes, off, n_l1);
    off += n_l1 * 4;
    let l2 = read_at(&bytes, off, n_l2);
    off += n_l2 * 4;
    let kan_flat = read_at(&bytes, off, n_kan);
    off += n_kan * 4;
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
    let n_tau = crate::gpu::evolution::N_CFC_CHANNELS;
    if base_off >= 20 {
        // Unlike historical optional-block files, v3 must contain the entire v2
        // fixed payload. A damaged checkpoint must not silently invent organs.
        let fixed_floats = crate::gpu::evolution::N_L1_WEIGHTS
            + crate::gpu::evolution::N_L2_WEIGHTS
            + n_kan
            + n_slow
            + n_tau
            + crate::gpu::evolution::N_SENSOR_GENES
            + 4 * crate::gpu::evolution::N_L1_NEURONS
            + 2 * crate::gpu::evolution::N_L2_NEURONS
            + 16
            + crate::gpu::hippocampus::N_HIP_GENES;
        if bytes.len() < base_off + fixed_floats * 4 {
            return None;
        }
    }

    let slow = if bytes.len() >= off + n_slow * 4 {
        let v = read_at(&bytes, off, n_slow);
        off += n_slow * 4;
        v
    } else {
        crate::engine::SlowBrain::new().all_weights()
    };
    let tau = if bytes.len() >= off + n_tau * 4 {
        let v = read_at(&bytes, off, n_tau);
        off += n_tau * 4;
        v
    } else {
        vec![crate::cfc::DECAY_DEFAULT; n_tau]
    };
    // SENSOR PHASE (step 1) — same optional pattern as slow/tau: a file written before
    // this field existed falls back to the default, which is the truth (that genome really
    // did run with today's hardcoded values, not with an invented one).
    let n_sensors = crate::gpu::evolution::N_SENSOR_GENES;
    let sensors = if bytes.len() >= off + n_sensors * 4 {
        let v = read_at(&bytes, off, n_sensors);
        off += n_sensors * 4;
        v
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
        let v = read_at(&bytes, off, n_l1 + n_l2);
        off += (n_l1 + n_l2) * 4;
        v
    } else {
        vec![0.0; n_l1 + n_l2]
    };
    let neuron_period_l1 = if bytes.len() >= off + n_l1 * 4 {
        let v = read_at(&bytes, off, n_l1);
        off += n_l1 * 4;
        v
    } else {
        vec![1.0; n_l1]
    };
    let neuron_threshold_l1 = if bytes.len() >= off + n_l1 * 4 {
        let v = read_at(&bytes, off, n_l1);
        off += n_l1 * 4;
        v
    } else {
        vec![0.0; n_l1]
    };
    let neuron_activation = if bytes.len() >= off + (n_l1 + n_l2) * 4 {
        let v = read_at(&bytes, off, n_l1 + n_l2);
        off += (n_l1 + n_l2) * 4;
        v
    } else {
        vec![0.0; n_l1 + n_l2]
    };
    // M4: the maestro's gates (optional; an old genome → zeros = neutral gates of 1.0).
    let gates_w = if bytes.len() >= off + 16 * 4 {
        let v = read_at(&bytes, off, 16);
        off += 16 * 4;
        v
    } else {
        vec![0.0; 16]
    };
    // C1: sector hippocampus (optional; an old genome → zeros = bias [0, 0, 0], i.e. the
    // brain that genome actually ran with). A file that stops before `gates_w` also stops
    // before this block: `off` did not advance, so the length test below fails too and
    // both fall back together — the blocks cannot get out of phase.
    let n_hip = crate::gpu::hippocampus::N_HIP_GENES;
    let hip = if bytes.len() >= off + n_hip * 4 {
        let v = read_at(&bytes, off, n_hip);
        off += n_hip * 4;
        v
    } else {
        vec![0.0; n_hip]
    };
    // FLY MAP: everything left in the file. It is the last block precisely so that its
    // length does not have to be known here — a reader without the map file still gets the
    // genes, and a file written before this block existed ends at `off` and yields an empty
    // block, which is the truth: that genome really did run without a fly map.
    //
    // A trailing remainder that is not a whole number of f32 is a damaged file, not a fly
    // block; it reads as empty rather than as a mis-strided vector.
    let fly = if bytes.len() > off && (bytes.len() - off) % 4 == 0 {
        read_at(&bytes, off, (bytes.len() - off) / 4)
    } else {
        Vec::new()
    };

    Some(Genome {
        l1,
        l2,
        kan_knots,
        slow,
        tau,
        sensors,
        neuron_alpha,
        neuron_period_l1,
        neuron_threshold_l1,
        neuron_activation,
        gates_w,
        hip,
        fly,
        body,
        organs,
        lifetime,
        organism,
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

#[cfg(test)]
mod organism_format_tests;
