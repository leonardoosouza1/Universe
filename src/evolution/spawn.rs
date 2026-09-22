use crate::config::iara::brain::{L1_NEURONS, SENSORY_WAVE_SIZE};
use crate::config::iara::vision::RETINA_TOTAL;
use crate::config::UniverseConfig;
use crate::simulation::Racer;
use crate::world::VoxelWorld;
use bevy::prelude::*;
use rand::prelude::*;
use std::io::Write;

use super::dna_genesis::{load_champion_dna, EliteDna, HypernetSeed, MorphoGenes};
use super::logs::open_report_log;
use super::settings::{DevSettings, RaceSettings};

/// Samples N(0,1) via Box-Muller from the thread RNG.
///
/// This formula was hand-written **eleven times** in this file, always identical,
/// always with the same `max(1e-7)` protecting the `ln(0)`. Eleven copies mean that
/// fixing one does not fix the others — that is exactly how a sign error in an
/// LCG survived seven copies and made 100% of the slow brain's weights be born
/// negative (see `src/rng.rs`).
///
/// Discards the second value of the pair (the variant in `gpu/evolution.rs` keeps the `spare`).
/// Here the call is sparse and per-weight, so keeping state between calls would require
/// carrying the pair along with the RNG with no measurable gain.
#[inline]
// The protected zone is disabled by setting its size to zero (see the constant and
// the measurement that motivated it), which makes `idx < PROTECTED_ZONE_SIZE`
// constant. The guards stay so the mechanism can be switched back on.
#[allow(clippy::absurd_extreme_comparisons)]
fn gauss(rng: &mut impl rand::Rng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-7);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Stride for weight indexing: weights[output * SWS + input_channel].
/// MUST be = SENSORY_WAVE_SIZE (input_dim of BitLinear). It was hardcoded 128,
/// causing misalignment when SENSORY_WAVE_SIZE changed to 223.
const SWS: usize = SENSORY_WAVE_SIZE;

// TIER 4.4 — NEUTRAL GENES (PROTECTED ZONE).
// Defined as 0: throttle occupies indices 0-222 and steer 223-399 (layout
// weights[output * SWS + input]). PROTECTED_ZONE_SIZE=400 froze
// 100% of throttle and 79% of steer — preventing evolution of the most
// important outputs and causing a plateau at 664m. A23 priming already provides
// initial values; evolution needs full freedom to improve from there.
// Deliberately zero, which makes every `idx < PROTECTED_ZONE_SIZE` guard below a
// no-op. Clippy reports those as `absurd_extreme_comparisons` and an audit read them
// as a bug — they are not. The guards are kept so the mechanism stays visible and can
// be switched back on by changing this one constant; the measured reason it is off is
// in the comment above. If you re-enable it, re-read that measurement first.
#[allow(clippy::absurd_extreme_comparisons)]
const PROTECTED_ZONE_SIZE: usize = 0;

/// Builds the instinct_layer weight vector pre-seeded with axioms A1–A15.
/// Used both in generation 0 (no pool/DNA) and periodically in later gens
/// to re-inject instincts lost to genetic drift.
// FIX: slow brain (32→8, 256 weights) stayed 100% zero for ~125 gens.
// Diagonal priming: each output k sums a band of 4 inputs → forces signal > 0
// and evolution fine-tunes from a functional base.

/// Random initialization for fast weights (L1 cortex).
/// Returns Vec<f32> with A24 sparse motor priming + random feature neurons.
///
/// KEY INSIGHT (A24): L1 output neurons 0-2 map to motor intent (throttle,
/// steer, brake) via L2's diagonal initialization. With N(0,0.1) random
/// weights, each motor neuron sums ~110 random ternary inputs that drown
/// the curvature priming signal (SNR ≈ 0.26). Fix: motor neurons 0-2
/// get ONLY priming weights (all other inputs zeroed). Feature neurons
/// 3-95 keep random weights for evolution to explore.
///
/// This makes curvature → brake response RELIABLE from gen 0 instead
/// of stochastic. Cars consistently brake before the hairpin at 754m.
///
/// Layout: weights[output * SWS + input_channel]
/// Motor neurons: 0=throttle, 1=steer, 2=brake (via L2 diagonal → final_intent).
// `0 * SWS` and the `idx < PROTECTED_ZONE_SIZE` guards below are deliberate: the row
// index names the motor neuron (and rhymes with `1 * SWS` / `2 * SWS`), and the
// protected zone is switched off by setting its size to zero, which makes those
// comparisons constant. Both read as defects to clippy and to a first-time reader;
// the reasons are at the constant definition above.
#[allow(clippy::erasing_op, clippy::absurd_extreme_comparisons)]
fn initialize_fast_weights_random() -> Vec<f32> {
    let mut rng = rand::rng();
    let mut weights = vec![0.0f32; SWS * L1_NEURONS];

    // Feature neurons 3..96: random N(0, 0.1) for evolutionary exploration.
    // Motor neurons 0..3: start at ZERO, only priming weights set below.
    for neuron in 3..L1_NEURONS {
        for ch in 0..SWS {
            let z = gauss(&mut rng);
            weights[neuron * SWS + ch] = z * 0.1;
        }
    }

    // MOTOR BIRTH MODE (see `neuro::Nascimento`).
    //
    // The priming below does TWO things at once: it makes the motor neuron SPARSE — which
    // solves the 0.26 SNR described in this function's comment — and it chooses WHICH channels it
    // listens to, which is injected human knowledge. The modes separate the two, so we can
    // ask whether evolution finds on its own the channels the human chose.
    match crate::neuro::nascimento() {
        crate::neuro::Nascimento::Primado => {} // proceeds to the hand priming, below
        crate::neuro::Nascimento::Denso => {
            // no sparsity: the motors become ordinary feature neurons. It is the floor
            // that the SNR comment describes as a drowned signal.
            for neuron in 0..3 {
                for ch in 0..SWS {
                    let z = gauss(&mut rng);
                    weights[neuron * SWS + ch] = z * 0.1;
                }
            }
            return weights;
        }
        crate::neuro::Nascimento::EsparsoAleatorio => {
            // same NUMBER of connections as the hand priming (3 on throttle, 10 on steer,
            // 2 on brake) and same magnitude range — only WHICH channels changes. This way the
            // comparison isolates the choice of content, not the quantity nor the strength.
            for neuron in 0..3 {
                for _ in 0..crate::neuro::n_conexoes_motor(neuron) {
                    let ch = (rng.random::<f32>() * SWS as f32) as usize % SWS;
                    let sinal = if rng.random::<f32>() < 0.5 { -1.0 } else { 1.0 };
                    weights[neuron * SWS + ch] = sinal * (3.0 + rng.random::<f32>() * 7.0);
                }
            }
            return weights;
        }
    }

    // === MOTOR NEURON 0: THROTTLE ===
    // Sparse: flow_alignment + curvature throttle-cut. No random noise.
    //
    // `0 * SWS` is deliberate and stays: the row index names the motor neuron and
    // lines up with `1 * SWS` (steer) and `2 * SWS` (brake) below. Clippy flags it
    // as `erasing_op`; folding it away would make the three blocks stop rhyming.
    weights[0 * SWS + 64] = 5.0; // flow_alignment → throttle boost
    weights[0 * SWS + 79] = -6.0; // curv 70m → cut throttle (curve ahead)
    weights[0 * SWS + 78] = -4.0; // curv 40m → cut throttle (curve imminent)

    // === MOTOR NEURON 1: STEER ===
    // Sparse: curvature probes at multiple distances + lane offset + dark center.
    weights[1 * SWS + 80] = 8.0; // curv 120m signed → steer preemptive
    weights[1 * SWS + 32] = 6.0; // curv 180m signed → steer far ahead
    weights[1 * SWS + 79] = 6.0; // curv 70m signed → steer near
    weights[1 * SWS + 78] = 4.0; // curv 40m signed → steer immediate
    weights[1 * SWS + 20] = 3.0; // curv 20m signed → steer reactive
    weights[1 * SWS + 81] = 3.0; // lane_offset → steer towards center
    for ch in 90..94 {
        weights[1 * SWS + ch] = 4.0; // dark center tracker → steer to road
    }

    // === MOTOR NEURON 2: BRAKE ===
    // Sparse: curvature probes only. No random noise, no ch20/ch28 (caused
    // over-braking at earlier curves → 661m plateau in v15).
    // ch79 (70m) overshoots at hairpin but works for WP24/32 curves.
    // ch78 (40m) sees the hairpin perfectly from 714m.
    // With sparse neuron, BOTH signals are clean → reliable curve braking.
    weights[2 * SWS + 79] = 10.0; // curv 70m → hard brake (WP24/32 curves)
    weights[2 * SWS + 78] = 8.0; // curv 40m → hard brake (CRITICAL for hairpin)

    weights
}

/// Spawns cars at index `[idx_start..idx_end)`. Allows spawning in blocks between frames
/// (avoids the freeze of spawning 2000 entities in 1 frame).
// The protected zone is disabled by setting its size to zero (see the constant and
// the measurement that motivated it), which makes `idx < PROTECTED_ZONE_SIZE`
// constant. The guards stay so the mechanism can be switched back on.
#[allow(clippy::absurd_extreme_comparisons)]
pub fn spawn_swarm(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    race: &mut RaceSettings,
    world: &VoxelWorld,
    dev: &DevSettings,
    config: &UniverseConfig,
    _lab: &crate::lab::LabParams,
    idx_start: usize,
    idx_end: usize,
) {
    let mut rng = rand::rng();
    let wing = meshes.add(Cuboid::new(0.6, 0.8, 2.0));
    let cockpit = meshes.add(Sphere::new(0.4));
    let names = [
        "Senna", "Hamilton", "Prost", "Schumi", "Alonso", "Vettel", "Max", "Lauda", "Leclerc",
        "Norris",
    ];
    let car_models = [
        "Porsche 911 GT3",
        "Ferrari 296 GT3",
        "McLaren 720S GT3",
        "Aston Martin Vantage",
        "Audi R8 LMS",
    ];

    let start_flow = world.get_flow_at(world.start_pos.x, world.start_pos.z);
    let _start_angle = start_flow.y.atan2(start_flow.x);

    let mut current_best_w = race.best_weights.clone();
    let mut current_best_s = race.best_scale;
    // Tier 6.3 LITE: morpho from the champion DNA (if loaded); default otherwise.
    let mut current_best_morpho: MorphoGenes = MorphoGenes::default();
    // Tier 6.4 LITE: hypernet seed from the champion DNA. Default = inactive (zero-cost).
    let mut current_best_hypernet: HypernetSeed = HypernetSeed::default();
    // v3.3: kan_knots inherited from CHAMPION.dna (default = tanh ramp).
    let mut current_best_kan_knots: [[f32; crate::kan::KAN_OUT_KNOTS]; 3] =
        crate::evolution::dna_genesis::default_kan_knots();

    // Mutation rate with exponential decay per generation: gradual convergence.
    // Floor at 0.005 keeps residual exploration even after 100+ generations.
    // v9: floor 0.005→0.012 to sustain exploration after 100+ gens.
    let effective_mutation_rate =
        (dev.mutation_rate * 0.97_f32.powi(race.generation as i32 - 1)).max(0.012) as f64;

    // Loads CHAMPION.dna (v3 with magic "IARA" or legacy v1/v2).
    // PER-WIDTH: tries CHAMPION_W{width}m.dna first (per-track memory,
    // does not unlearn when switching tracks), generic CHAMPION.dna fallback.
    if current_best_w.is_none() {
        let cw = config.current_track().track_width as i32;
        let per_width_path = format!("CHAMPION_W{}m.dna", cw);
        let dna_loaded =
            load_champion_dna(&per_width_path).or_else(|| load_champion_dna("CHAMPION.dna"));
        if let Some(dna) = dna_loaded {
            current_best_w = Some(if let Some(ref craw) = dna.cortex_raw {
                craw.iter().map(|&w| w as f32 / 63.5).collect()
            } else {
                dna.fast_weights.iter().map(|&w| w as f32 / 63.5).collect()
            });
            // V7: load cortex_l2 weights (Layer 2 deep). Default zeros if absent
            // (pre-V7 DNAs) — cortex_l2 initializes to zeros, mutation fills in.
            if let Some(ref l2) = dna.cortex_l2_raw {
                race.best_l2_weights = Some(l2.iter().map(|&w| w as f32 / 63.5).collect());
            }
            current_best_s = dna.fast_scale;

            // SlowBrain: prefer SLWF raw f32 (lossless), fallback to ternary i8.
            if let Some(ref sf) = dna.slow_f32 {
                race.best_slow_weights = Some(sf.clone());
            } else if let Some(ref slow) = dna.slow_weights {
                let slow_f32: Vec<f32> = slow.iter().map(|&b| b as f32 / 127.0).collect();
                race.best_slow_weights = Some(slow_f32);
            }

            // Tier 6.3 LITE: absorbs serialized morpho genes.
            current_best_morpho = dna.morpho;
            // Tier 6.4 LITE: absorbs hypernet seed (default inactive for old files).
            current_best_hypernet = dna.hypernet;
            // v3.3: absorbs KAN knots (preserves the learned spline between runs).
            current_best_kan_knots = dna.kan_knots;
            // v5: absorbs Trinity gate + reflex weights (TRIN chunk). Restores the swarm
            // learned before reboot — without this, gate/reflex re-init uniform and
            // lose the safety net's learning.
            if let Some(ref t) = dna.trinity {
                race.best_gate_weights = Some(t.gate_weights.to_vec());
                race.best_gate_bias = Some(t.gate_bias.to_vec());
                // Dequantize i8 TRIN bytes → f32 runtime weights. Mapping i8/127 covers
                // [-1.0, 1.0]. Cars trained pre-V5 had ternary {-1,0,+1} which maps cleanly.
                let mut rw: Vec<f32> = Vec::with_capacity(96);
                for &b in &t.reflex_steer {
                    rw.push(b as f32 / 127.0);
                }
                for &b in &t.reflex_throttle {
                    rw.push(b as f32 / 127.0);
                }
                for &b in &t.reflex_brake {
                    rw.push(b as f32 / 127.0);
                }
                race.best_reflex_weights = Some(rw);
            }
            // If the DNA was trained on another seed/width, logs a warning and discards champion_distance
            // (otherwise the new car can never be crowned on a different track).
            if dna.version >= 3 {
                let same_track = dna.seed as u64 == race.seed
                    && (dna.track_width - config.current_track().track_width).abs() < 0.1;
                if !same_track {
                    race.champion_distance = 0.0;
                    if let Some(mut file) = open_report_log() {
                        let _ = writeln!(
                            file,
                            "⚠️ CHAMPION.dna came from another track (seed {} width {:.1}) — threshold reset",
                            dna.seed, dna.track_width
                        );
                    }
                }
                // V6: do NOT pump race.generation to dna.generation — evolution state
                // (elite_pool, fitness_history, env transitions) starts empty on UI
                // reload, a pumped gen caused mismatch (rain transitions, time dilation
                // skip_n=2 at gen≥10, etc.) that degraded the loaded DNA.
            }
        }
    }

    // Helper: tournament k=3 selects 1 parent among k random competitors from the pool.
    let tournament = |pool: &[EliteDna], rng: &mut rand::rngs::ThreadRng| -> usize {
        let mut best_idx = rng.random_range(0..pool.len());
        for _ in 1..3 {
            let cand = rng.random_range(0..pool.len());
            if pool[cand].fitness > pool[best_idx].fitness {
                best_idx = cand;
            }
        }
        best_idx
    };

    // Cluster-aware tournament p2 picker: runs standard tournament and rejects
    // any candidate whose niching cluster matches p1's. After 5 failed attempts
    // the fallback accepts a same-cluster parent (pool may be dominated by one
    // niche early on — better to reproduce than stall). Cluster ids are parallel
    // to race.elite_pool; truncated to effective_pool to stay within index range.
    let pick_p2_diff_cluster = |pool: &[EliteDna],
                                clusters: &[usize],
                                p1: usize,
                                rng: &mut rand::rngs::ThreadRng|
     -> usize {
        let c1 = clusters.get(p1).copied().unwrap_or(usize::MAX);
        for _ in 0..5 {
            let cand = tournament(pool, rng);
            if cand == p1 {
                continue;
            }
            let cc = clusters.get(cand).copied().unwrap_or(usize::MAX);
            if cc != c1 {
                return cand;
            }
        }
        // Fallback: same-cluster allowed after 5 rejections.
        let cand = tournament(pool, rng);
        if cand == p1 && pool.len() > 1 {
            (p1 + 1) % pool.len()
        } else {
            cand
        }
    };

    // Helper: BLOCK-WISE crossover (Tier 4.1 — sexual reproduction).
    // Swaps whole blocks between parents instead of byte-by-byte. Preserves functional
    // coherence: each block = one neuron's input row (fast: 128 inputs →
    // block_size=128 gives the whole row; here we use 64 for half a row, keeping
    // reasonable granularity without over-fragmenting). For slow_weights we use
    // block_size=32 (half of the 8 outputs × 32 inputs).
    let crossover_block =
        |a: &[f32], b: &[f32], block_size: usize, rng: &mut rand::rngs::ThreadRng| -> Vec<f32> {
            let n = a.len();
            let mut child = vec![0.0f32; n];
            let mut i = 0;
            while i < n {
                let end = (i + block_size).min(n);
                let from = if rng.random_bool(0.5) { a } else { b };
                for j in i..end {
                    child[j] = from[j];
                }
                i = end;
            }
            child
        };

    // SPECIES SIMILARITY (f32 weights): fraction of positions where signs match.
    let species_similarity = |a: &[f32], b: &[f32]| -> f32 {
        let n = a.len().min(b.len()).max(1);
        let mut agree = 0usize;
        for i in 0..n {
            let sa = a[i].signum();
            let sb = b[i].signum();
            if sa == sb || (sa == 0.0 && sb == 0.0) {
                agree += 1;
            }
        }
        agree as f32 / n as f32
    };

    // Helper: BURST MUTATION per input channel.
    // Fast layer layout: weights[out * 128 + in], 64 outputs × 128 inputs.
    // Burst = picks an input channel c and replaces ALL 64 connections
    // (for each output) with random values in [-1, 1]. Breaks BitNet neutrality:
    // point mutation in a ternary layer has high probability of becoming 0 after quantize —
    // burst forces coherent variation of an entire sensor.
    let burst_mutate_input_channel =
        |weights: &mut Vec<f32>, scar_mask_union: &[u8], rng: &mut rand::rngs::ThreadRng| {
            let n_outputs = L1_NEURONS;
            let n_inputs = SWS;
            let c = rng.random_range(0..n_inputs);
            for out in 0..n_outputs {
                let idx = out * n_inputs + c;
                if idx < weights.len() {
                    // TIER 4.4: protected zone is exempt from burst mutation too.
                    if idx < PROTECTED_ZONE_SIZE {
                        continue;
                    }
                    // TIER 6.6: respects scars — does not touch locked weights.
                    if scar_mask_union.get(idx).copied().unwrap_or(0) > 0 {
                        continue;
                    }
                    weights[idx] = rng.random_range(-1.0..1.0);
                }
            }
        };

    // REMOVED: axiom_decay function and axiom_strength logic.
    // Gen 0 forward: purely random initialization, no hardcoded steering biases.
    // Evolution learns from scratch via mutation + selection.

    // Tier 6.4 LITE: hypernet seed mutation 1x per gen (first batch).
    // 2% chance of flipping `active`; if active, gaussian noise σ=0.05 on the coeffs.
    // Mutation is runtime-only in this LITE slice: persistence via .dna requires a
    // follow-up in pipeline.rs (out of scope). Zero-cost when inactive.
    if idx_start == 0 {
        current_best_hypernet.mutate(|| rng.random::<f32>());
    }
    // Expose for pipeline save (Tier 6.3/6.4 persistence).
    race.current_morpho = current_best_morpho;
    race.current_hypernet = current_best_hypernet;

    let pool_size = race.elite_pool.len();
    // Snapshot niching clusters (parallel to elite_pool). Empty until pipeline
    // populates it on first post-race pass; gen 0 spawn hits this before any
    // clustering, so default is empty and pick_p2_diff_cluster degrades cleanly
    // to standard tournament.
    let elite_clusters: Vec<usize> = race.elite_clusters.clone();

    // Tier 4.2: accumulates inheritance sample count + sums for the aggregate log at the end.
    let mut epi_inherited_count: usize = 0;
    let mut epi_cortisol_sum: f32 = 0.0;
    let mut epi_dopamine_sum: f32 = 0.0;
    let mut epi_curiosity_sum: f32 = 0.0;

    for i in idx_start..idx_end.min(config.swarm_count) {
        let mut engine = crate::engine::SovereignEngine::new();

        // --- HALL OF FAME (CHAMPION FALLBACK) ---
        // Slot 0: ALWAYS loads the CHAMPION DNA if it exists. Immune to generational drift
        // (even if the entire elite_pool is bad, the HoF preserves the best of all time).
        // v13: 1 pure champion clone (was 3). 3 clones caused 15-18% crash rate from
        // collisions on narrow track despite denoise seeding. 1 clone had 4% crash rate.
        // Denoise seeding still provides stochastic robustness for the single clone.
        let slot_is_champion = i == 0 && current_best_w.is_some();

        // Denoising phase per car: gen-hash XOR slot for diversity.
        // All cars (including champion) get different noise each gen, testing
        // robustness across noise patterns. This is what enabled the 2450m breakthrough.
        let gen_hash = (race.generation as u32).wrapping_mul(2654435761);
        let denoise_seed = (i as u32).wrapping_mul(7919) ^ gen_hash;
        engine.slow_brain.cfc.seed_denoise(denoise_seed);
        engine.cfc_state.seed_denoise(denoise_seed.wrapping_add(13));
        // Tier 4.2: partial epigenetic inheritance (10%) — if there are parent(s).
        // Filled in the reproduction branches; None = priming/boot organism.
        let mut epigenetic_parents: Option<(f32, f32, f32)> = None;
        // Tier 6.3 LITE: morpho of the new racer. Filled in the branches according to source
        // (champion / elite clone / parents crossover / priming). Mutated and clamped.
        let mut morpho_this: MorphoGenes = MorphoGenes::default();

        // SIMPLE ELITISM v2: slot 0 = pure champion, slots 1..(N-15) = champion + mutation,
        // last 15 slots = PRIMING REFRESH (fresh DNA) to avoid monoculture.
        // User reported: gen 2 ALL crash at the same curve (clones of the champion's bug).
        // Priming refresh provides diversity — if the champion is stuck in a local minimum,
        // priming cars explore alternatives.
        let priming_slot = i >= config.swarm_count.saturating_sub(5);
        #[cfg(feature = "simple_elitism")]
        let simple_mut_slot = !slot_is_champion && !priming_slot && current_best_w.is_some();
        #[cfg(not(feature = "simple_elitism"))]
        let simple_mut_slot = false;

        if priming_slot {
            // Fresh random initialization N(0, 0.1) for diversity.
            // No hardcoded steering — purely learned evolution.
            let fast_random = initialize_fast_weights_random();
            engine.instinct_layer.quantize(&fast_random);
            engine.instinct_layer.set_scale(1.0);
            // NOTE: SlowBrain now uses stateful CfC + Dense architecture.
            // It is initialized in SovereignEngine::new() and not mutated in spawn.
        } else if slot_is_champion || simple_mut_slot {
            if let Some(ref champion) = current_best_w {
                engine.instinct_layer.quantize(champion);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref l2) = race.best_l2_weights {
                    let n = l2.len().min(engine.cortex_l2.weights.len());
                    engine.cortex_l2.weights[..n].copy_from_slice(&l2[..n]);
                }
                // CHAMPION VERIFY: print a simple checksum of the DNA loaded into slot 0.
                // User asked for validation that the champion DNA really loads across gens.
                if i == 0 {
                    let cksum: f32 = champion.iter().take(64).sum();
                    let l2_cksum: f32 = race
                        .best_l2_weights
                        .as_ref()
                        .map(|v| v.iter().take(64).sum::<f32>())
                        .unwrap_or(0.0);
                    let nonzero_l1 = champion.iter().filter(|&&w| w.abs() > 0.01).count();
                    println!(
                        "✓ SLOT 0 CHAMPION gen={} L1_sum={:.3} L2_sum={:.3} L1_active={} (DNA loaded)",
                        race.generation, cksum, l2_cksum, nonzero_l1
                    );
                }
                // Hypernet DISABLED for f32 cortex — expansion was ternary {-1,0,1}/127
                // ≈ ±0.008 magnitude that overwrote trained f32 weights (~±2.0),
                // destroying the network at spawn. Hypernet is left for V2 with a real f32 basis.
                let _ = current_best_hypernet;
                if let Some(ref slow_w) = race.best_slow_weights {
                    engine.slow_brain.set_all_weights(slow_w);
                }
            }
            // Tier 6.3 LITE: champion preserves exact morpho (no mutation in the HoF).
            morpho_this = current_best_morpho;

            // simple_elitism: slot != 0 undergoes soft gaussian mutation (f32 runtime).
            // Box-Muller σ=0.05 — small adjustments instead of disruptive flips.
            if simple_mut_slot {
                let rate_boosted = (effective_mutation_rate * 2.0).min(0.25);
                // STRATIFIED mutation: top slots stay close to the champion (nearly
                // pure clones), bottom slots explore. User report: gen N+1 majority did
                // not replicate the gen N champion. Before, uniform σ 0.06+ destroyed inheritance.
                let g = race.generation.max(1);
                let base_decay = 0.04_f32 + 0.06_f32 * 0.88_f32.powi((g as i32 - 1).max(0));
                // The EXPLORER band (doubled σ) starts where `fracao_exploradora` says.
                // It used to be fixed at slot 30 of 60, i.e. HALF the population — and measured
                // in the reports: the population MEDIAN is zero meters, with `wall_mean = 0,00s`.
                // It is neither a crash nor a wall; it is mutation destroying the genome. The
                // effective population was 30, not 60. Now which fraction pays off can be measured.
                let n_pop = config.swarm_count.max(1);
                let ini_explora =
                    ((1.0 - crate::neuro::fracao_exploradora()) * n_pop as f32) as usize;
                let sigma_cortex = if i >= ini_explora {
                    base_decay * 2.0 // explore (σ ≈ 0.08-0.20)
                } else if i * 2 >= ini_explora {
                    base_decay * 1.0 // medium (σ ≈ 0.04)
                } else if i * 6 >= ini_explora {
                    base_decay * 0.5 // small (σ ≈ 0.02)
                } else {
                    base_decay * 0.15 // near-exact clones (σ ≈ 0.006)
                };
                for w in engine.instinct_layer.weights.iter_mut() {
                    if rng.random_bool(rate_boosted) {
                        let z = gauss(&mut rng);
                        *w = (*w + sigma_cortex * z).clamp(-10.0, 10.0);
                    }
                }
                for w in engine.cortex_l2.weights.iter_mut() {
                    if rng.random_bool(rate_boosted) {
                        let z = gauss(&mut rng);
                        *w = (*w + sigma_cortex * z).clamp(-10.0, 10.0);
                    }
                }
                // cortex_l3 removed entirely on 2026-08-14 (P1.9): it was never in the
                // forward pass nor in any disk format — see docs/PLANO_EVOLUCAO.md.
                // SlowBrain mutation: Gaussian σ scaled like cortex, clamped [-10, 10].
                // CfC state resets on spawn — only Dense1+Dense2 weights are heritable.
                {
                    let slow_all = engine.slow_brain.all_weights();
                    let mut slow_mut = slow_all.clone();
                    let sigma_slow = sigma_cortex * 1.5; // slightly higher σ — fewer weights, needs more exploration
                    for w in slow_mut.iter_mut() {
                        if rng.random_bool(rate_boosted) {
                            let z = gauss(&mut rng);
                            *w = (*w + sigma_slow * z).clamp(-10.0, 10.0);
                        }
                    }
                    engine.slow_brain.set_all_weights(&slow_mut);
                }
                // Morpho lightly jittered — keeps proximity to the champion
                // (refactor without an extra MorphoGenes::jitter dep).
            }

            // ---- Trinity: copy champion gate + reflex into this engine ----
            if let Some(ref gw) = race.best_gate_weights {
                let n = gw.len().min(engine.gate.weights.len());
                engine.gate.weights[..n].copy_from_slice(&gw[..n]);
            }
            if let Some(ref gb) = race.best_gate_bias {
                let n = gb.len().min(engine.gate.bias.len());
                engine.gate.bias[..n].copy_from_slice(&gb[..n]);
            }
            if let Some(ref rw) = race.best_reflex_weights {
                // SpecialistBrain has 3 FloatLinears: steer (32), throttle (32), brake (32) = 96 f32 weights.
                let steer_len = engine.reflex.steer.weights.len();
                let throttle_len = engine.reflex.throttle.weights.len();
                let brake_len = engine.reflex.brake.weights.len();
                let total = steer_len + throttle_len + brake_len;
                if rw.len() >= total {
                    engine
                        .reflex
                        .steer
                        .weights
                        .copy_from_slice(&rw[0..steer_len]);
                    engine
                        .reflex
                        .throttle
                        .weights
                        .copy_from_slice(&rw[steer_len..steer_len + throttle_len]);
                    engine.reflex.brake.weights.copy_from_slice(
                        &rw[steer_len + throttle_len..steer_len + throttle_len + brake_len],
                    );
                }
            }

            // ---- Trinity: mutate gate + reflex on simple_mut_slot ----
            if simple_mut_slot {
                let rate_boosted = (effective_mutation_rate * 2.0).min(0.25);
                let rate = (rate_boosted * 0.5) as f64; // half rate vs cortex — gate/reflex more sensitive
                for w in engine.gate.weights.iter_mut() {
                    if rng.random_bool(rate) {
                        *w += (rng.random::<f32>() - 0.5) * 0.2;
                    }
                }
                for b in engine.gate.bias.iter_mut() {
                    if rng.random_bool(rate) {
                        *b += (rng.random::<f32>() - 0.5) * 0.1;
                    }
                }
                // Reflex weights are f32 (FloatLinear). Mutate smooth gaussian — small
                // adjustments instead of a disruptive ternary flip. σ=0.05, clamp [-10.0, 10.0].
                for layer in [
                    &mut engine.reflex.steer,
                    &mut engine.reflex.throttle,
                    &mut engine.reflex.brake,
                ] {
                    for w in layer.weights.iter_mut() {
                        if rng.random_bool(rate) {
                            // Box-Muller approx via 2 uniforms; cheap noise.
                            let z = gauss(&mut rng);
                            *w = (*w + 0.05 * z).clamp(-10.0, 10.0);
                        }
                    }
                }
            }
            // ---- end Trinity ----
        } else if pool_size >= 2 {
            // --- DIVERSIFIED REPRODUCTION ON THE GRID (30 starting slots) ---
            // Previous bottleneck: only 5 grid spawns × 2 elite clones = 2 DNAs tested
            // from the start line per gen. A 935m DNA sat in the distributed population
            // without ever being validated on "start from zero". Now we spread the whole
            // pool across the 30 grid slots:
            //   Slot 0:      CHAMPION (already handled above)
            //   Slot 1-10:   top-10 elite clones (each elite tested on the grid)
            //   Slot 11-20:  PRIMING REFRESH (fresh instincts at the start line)
            //   Slot 21-29:  tournament crossover (diverse children on the grid)
            //   Slot 30+:    tournament crossover (distributed, track coverage)
            let elite_clones = pool_size.min(10);
            let refresh_slots = 10;
            let elite_range_end = 1 + elite_clones; // [1, 1+elite_clones)

            if i >= 1 && i < elite_range_end {
                let elite = &race.elite_pool[i - 1];
                let exact: Vec<f32> = elite.fast_weights.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(elite.fast_scale);

                engine.slow_brain.set_all_weights(&elite.slow_weights);
                engine.cortex_l2.set_weights(&elite.cortex_l2);
                if !elite.perception_weights.is_empty() {
                    engine
                        .retina_perception
                        .load_weights(&elite.perception_weights);
                }
                // Tier 4.2: clone inherits 10% of the single parent's mean hormonal state.
                epigenetic_parents = Some((
                    elite.cortisol_mean,
                    elite.dopamine_mean,
                    elite.curiosity_mean,
                ));
                // Tier 6.3 LITE: clone inherits the elite's morpho + light mutation.
                morpho_this = elite.morpho;
                morpho_this.mutate(|| rng.random::<f32>());
            } else if i < elite_range_end + refresh_slots {
                // PRIMING REFRESH: fresh random N(0, 0.1) for diversity.
                // No axiom bias — pure evolution from scratch.
                let fast_random = initialize_fast_weights_random();
                engine.instinct_layer.quantize(&fast_random);
                engine.instinct_layer.set_scale(1.0);
                // SlowBrain initialized in SovereignEngine::new()
            } else {
                // Tier 4.1 — MATE PREFERENCE.
                // Selects p1 via tournament (top pool). Picks p2 weighted by
                // species similarity:
                //   similarity > 0.7 (same species): weight 0.7 (specialization)
                //   similarity <= 0.7 (different species): weight 0.3 (diversity)
                // Effective pool: top-10 of the elite_pool (already sorted by fitness).
                let effective_pool = pool_size.min(10);
                let p1 = tournament(&race.elite_pool[..effective_pool], &mut rng);
                // NICHING: p2 via cluster-aware rejection tournament (different
                // BehaviorFingerprint cluster from p1, up to 5 attempts, fallback
                // to same-cluster). If clusters not yet populated (gen 1 boot),
                // fall back to legacy species-similarity mate preference.
                let clusters_slice: &[usize] = if elite_clusters.len() >= effective_pool {
                    &elite_clusters[..effective_pool]
                } else {
                    &[]
                };
                let p2 = if !clusters_slice.is_empty() {
                    pick_p2_diff_cluster(
                        &race.elite_pool[..effective_pool],
                        clusters_slice,
                        p1,
                        &mut rng,
                    )
                } else {
                    let parent_a_ref = &race.elite_pool[p1];
                    let mut weights_pref: Vec<f32> = Vec::with_capacity(effective_pool);
                    for j in 0..effective_pool {
                        if j == p1 {
                            weights_pref.push(0.0);
                            continue;
                        }
                        let sim = species_similarity(
                            &parent_a_ref.fast_weights,
                            &race.elite_pool[j].fast_weights,
                        );
                        let w = if sim > 0.7 { 0.7 } else { 0.3 };
                        weights_pref.push(w);
                    }
                    let total_w: f32 = weights_pref.iter().sum();
                    if total_w > 1e-6 {
                        let mut target = rng.random_range(0.0..total_w);
                        let mut pick = 0usize;
                        for (j, &w) in weights_pref.iter().enumerate() {
                            target -= w;
                            if target <= 0.0 {
                                pick = j;
                                break;
                            }
                            pick = j;
                        }
                        pick
                    } else if pool_size > 1 {
                        (p1 + 1) % pool_size
                    } else {
                        p1
                    }
                };
                let parent_a = &race.elite_pool[p1];
                let parent_b = &race.elite_pool[p2];

                // Tier 4.1 stats: log the mate preference used (1x per gen, first crossover slot).
                if i == elite_range_end + refresh_slots {
                    let sim = species_similarity(&parent_a.fast_weights, &parent_b.fast_weights);
                    let kind = if sim > 0.7 {
                        "same-species"
                    } else {
                        "cross-species"
                    };
                    if let Some(mut file) = open_report_log() {
                        let _ = writeln!(
                            file,
                            "[MATE] gen {} pool={} p1={} p2={} sim={:.2} ({})",
                            race.generation, effective_pool, p1, p2, sim, kind
                        );
                    }
                }

                let mut crossed_fast =
                    crossover_block(&parent_a.fast_weights, &parent_b.fast_weights, 64, &mut rng);
                // TIER 4.4 — PROTECTED ZONE: first 819 indices (10%) are copied
                // straight from parent_a, no block-alternate. Preserves ancestral axioms.
                for idx in 0..PROTECTED_ZONE_SIZE.min(crossed_fast.len()) {
                    crossed_fast[idx] = parent_a.fast_weights[idx] as f32;
                }
                // TIER 6.6 — WEIGHT SCARS: a weight with scar TTL>0 in EITHER parent is locked from mutation.
                // Union: if parent_a.scar_mask[i] > 0 OR parent_b.scar_mask[i] > 0 → protect.
                // The value inherited in the crossover already stuck; mutation is what we skip.
                let scar_a = &parent_a.scar_mask;
                let scar_b = &parent_b.scar_mask;
                let mut mutated_fast: Vec<f32> = crossed_fast
                    .into_iter()
                    .enumerate()
                    .map(|(idx, w)| {
                        // TIER 4.4: protected zone exempt from mutation.
                        if idx < PROTECTED_ZONE_SIZE {
                            return w;
                        }
                        let scarred = scar_a.get(idx).copied().unwrap_or(0) > 0
                            || scar_b.get(idx).copied().unwrap_or(0) > 0;
                        if !scarred && rng.random_bool(effective_mutation_rate) {
                            // Gaussian mutation σ=0.05 for continuous f32 weights
                            let z = gauss(&mut rng);
                            (w + z * 0.05).clamp(-10.0, 10.0)
                        } else {
                            w
                        }
                    })
                    .collect();
                // 20% of the crossover slots get a burst per input channel (on top of point mutation).
                if rng.random_bool(0.20) {
                    // Union of the two parents' scars (same logic as the point mutation above).
                    let scar_union: Vec<u8> = scar_a
                        .iter()
                        .zip(scar_b.iter())
                        .map(|(&a, &b)| a.max(b))
                        .collect();
                    burst_mutate_input_channel(&mut mutated_fast, &scar_union, &mut rng);
                }
                engine.instinct_layer.quantize(&mutated_fast);
                engine
                    .instinct_layer
                    .set_scale((parent_a.fast_scale + parent_b.fast_scale) * 0.5);

                // Perception: uniform crossover + gaussian mutation
                if !parent_a.perception_weights.is_empty()
                    && !parent_b.perception_weights.is_empty()
                {
                    let pw: Vec<f32> = parent_a
                        .perception_weights
                        .iter()
                        .zip(parent_b.perception_weights.iter())
                        .map(|(&a, &b)| {
                            let base = if rng.random_bool(0.5) { a } else { b };
                            if rng.random_bool(effective_mutation_rate) {
                                let z = gauss(&mut rng);
                                (base + z * 0.05).clamp(-10.0, 10.0)
                            } else {
                                base
                            }
                        })
                        .collect();
                    engine.retina_perception.load_weights(&pw);
                }
                // SlowBrain: crossover + Gaussian mutation (same as perception path).
                if !parent_a.slow_weights.is_empty() && !parent_b.slow_weights.is_empty() {
                    let sw: Vec<f32> = parent_a
                        .slow_weights
                        .iter()
                        .zip(parent_b.slow_weights.iter())
                        .map(|(&a, &b)| {
                            let base = if rng.random_bool(0.5) { a } else { b };
                            if rng.random_bool(effective_mutation_rate) {
                                let z = gauss(&mut rng);
                                (base + z * 0.08).clamp(-10.0, 10.0) // σ=0.08, slightly higher for exploration
                            } else {
                                base
                            }
                        })
                        .collect();
                    engine.slow_brain.set_all_weights(&sw);
                }
                // Tier 4.2: MEAN hormonal inheritance from the 2 parents (sexual reproduction).
                let cort = (parent_a.cortisol_mean + parent_b.cortisol_mean) * 0.5;
                let dopa = (parent_a.dopamine_mean + parent_b.dopamine_mean) * 0.5;
                let curi = (parent_a.curiosity_mean + parent_b.curiosity_mean) * 0.5;
                epigenetic_parents = Some((cort, dopa, curi));
                // Tier 6.3 LITE: morpho = mean of the parents + gaussian mutation.
                let pa_m = parent_a.morpho;
                let pb_m = parent_b.morpho;
                morpho_this = MorphoGenes {
                    mass: (pa_m.mass + pb_m.mass) * 0.5,
                    wheelbase: (pa_m.wheelbase + pb_m.wheelbase) * 0.5,
                    downforce: (pa_m.downforce + pb_m.downforce) * 0.5,
                    grip_mult: (pa_m.grip_mult + pb_m.grip_mult) * 0.5,
                }
                .sanitized();
                morpho_this.mutate(|| rng.random::<f32>());
            }
        } else if let Some(ref best) = current_best_w {
            // Tier 6.3 LITE: fallback without pool → inherits the champion's morpho (+ light mutation if i>0).
            morpho_this = current_best_morpho;
            if i >= 3 {
                morpho_this.mutate(|| rng.random::<f32>());
            }
            // --- FALLBACK: DNA from CHAMPION.dna without pool (first generation after boot) ---
            // Slots 0,1,2 = exact champion clones (3x elitism buffer). Slots 3+ mutate.
            if i < 3 {
                let exact: Vec<f32> = best.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref slow_w) = race.best_slow_weights {
                    engine.slow_brain.set_all_weights(slow_w);
                }
                if let Some(ref l2) = race.best_l2_weights {
                    engine.cortex_l2.set_weights(l2);
                }
            } else {
                let mutated: Vec<f32> = best
                    .iter()
                    .enumerate()
                    .map(|(idx, &w)| {
                        // TIER 4.4: protected zone exempt from mutation.
                        if idx < PROTECTED_ZONE_SIZE {
                            return w as f32;
                        }
                        if rng.random_bool(effective_mutation_rate) {
                            // Gaussian mutation σ=0.05
                            let z = gauss(&mut rng);
                            (w as f32 + z * 0.05).clamp(-10.0, 10.0)
                        } else {
                            w as f32
                        }
                    })
                    .collect();
                engine.instinct_layer.quantize(&mutated);
                engine.instinct_layer.set_scale(current_best_s);
                // NOTE: SlowBrain not mutated. Uses fresh init from SovereignEngine::new()
            }
        } else {
            // COLD START: no champion, no elite pool. ALL cars get A23 priming.
            // Without this, only priming_slot (5 cars) had curvature instincts —
            // the other 25 were pure random N(0,0.1) and couldn't navigate curves.
            let fast_random = initialize_fast_weights_random();
            engine.instinct_layer.quantize(&fast_random);
            engine.instinct_layer.set_scale(1.0);
        }

        // Tier 4.2: seeds the child's metabolism with 10% of the parent(s) hormonal state.
        // Creates lineages with "trauma" (high cortisol) or "calm" (low cortisol).
        if let Some((p_cort, p_dopa, p_curi)) = epigenetic_parents {
            engine.metabolism.seed_epigenetic(p_cort, p_dopa, p_curi);
            epi_inherited_count += 1;
            epi_cortisol_sum += p_cort;
            epi_dopamine_sum += p_dopa;
            epi_curiosity_sum += p_curi;
        }

        let color = if i == 0 && current_best_w.is_some() {
            Color::srgb(0.0, 1.0, 0.8)
        } else {
            Color::srgb(
                rng.random::<f32>(),
                rng.random::<f32>(),
                rng.random::<f32>(),
            )
        };
        let racer_mat = materials.add(StandardMaterial {
            base_color: color,
            emissive: LinearRgba::from(color) * 4.0,
            perceptual_roughness: 0.1,
            metallic: 0.9,
            ..default()
        });

        // --- SPAWN DISTRIBUTED ALONG THE TRACK (Andrychowicz et al, 2017) ---
        // Cars 0..29 (champion + 10 elites + 10 priming + 9 crossover) start at the
        // start line — 30 independent attempts of the good DNA, enough for at least
        // one to survive the start and prove transferable skill.
        // Cars 30+ spawn at points spread along the track (coverage).
        // Fitness uses NET progress (track_progress - spawn_progress), so cars
        // born in a curve that survived 30m have PROVEN cornering ability.
        // This breaks the deceptive plateau of "only straights give high fitness".
        // simple_elitism: ALL on the grid. No distributed spawn.
        // Each car is champion+mutation, all start from zero = fair comparison.
        #[cfg(feature = "simple_elitism")]
        const GRID_SPAWNS: usize = usize::MAX;
        #[cfg(not(feature = "simple_elitism"))]
        const GRID_SPAWNS: usize = 30; // rollback v7 (grid 50 regressed)
        let (spawn_pos, spawn_rot, spawn_progress) = if i < GRID_SPAWNS {
            // Grid spawn along the waypoints (follows the track). Each row backs off 6m
            // in track arc, lateral spread per col keeps it within the road width.
            // Before: moved straight ahead from start_pos → Catmull-Rom curves put cars
            // off the track. Now: takes the waypoint segment N behind and positions there.
            #[cfg(feature = "simple_elitism")]
            const GRID_COLS: usize = 4;
            #[cfg(not(feature = "simple_elitism"))]
            const GRID_COLS: usize = 2;
            let row = i / GRID_COLS;
            let col = i % GRID_COLS;
            let col_offset = col as f32 - (GRID_COLS as f32 - 1.0) / 2.0;
            let back_dist = row as f32 * 6.0; // meters backwards along the track

            // Locates the waypoint of the start_pos segment.
            let n_wp = world.waypoints_world.len();
            let start_progress = world.start_progress;
            let lap = world.lap_distance.max(1.0);
            // Target progress (behind the start, with wrap).
            let target_progress = (start_progress - back_dist + lap) % lap;
            // Finds the segment covering target_progress via waypoint_arcs.
            let mut seg = 0usize;
            for k in 0..n_wp.saturating_sub(1) {
                if world.waypoint_arcs[k] <= target_progress
                    && target_progress < *world.waypoint_arcs.get(k + 1).unwrap_or(&lap)
                {
                    seg = k;
                    break;
                }
            }
            let wp_a = world.waypoints_world[seg];
            let wp_b = world.waypoints_world[(seg + 1).min(n_wp - 1)];
            let seg_arc_a = world.waypoint_arcs[seg];
            let seg_arc_b = *world.waypoint_arcs.get(seg + 1).unwrap_or(&lap);
            let t = if seg_arc_b > seg_arc_a {
                ((target_progress - seg_arc_a) / (seg_arc_b - seg_arc_a)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let center_2d = wp_a + (wp_b - wp_a) * t;
            let dir_2d = (wp_b - wp_a).normalize_or_zero();
            let perp_2d = Vec2::new(-dir_2d.y, dir_2d.x);
            let pos_2d = center_2d + perp_2d * (col_offset * 3.0);
            let p = Vec3::new(pos_2d.x, world.start_pos.y, pos_2d.y);
            let angle = dir_2d.y.atan2(dir_2d.x);
            // GRID spawn: spawn_progress=0 marks it as "start line" for the is_grid_spawn
            // crown gate. Net progress later subtracts 0 → pure track_progress.
            // (Before, target_progress confused the gate and blocked DNA crowning).
            let _ = target_progress;
            (p, Quat::from_rotation_y(-angle), 0.0_f32)
        } else {
            // Distributes along the waypoints. Fraction = 25%, 50%, 75%, 87%... in a circular band.
            let distributed_idx = (i - GRID_SPAWNS) as f32;
            let total_distributed = (config.swarm_count - GRID_SPAWNS).max(1) as f32;
            let fraction = 0.15 + (distributed_idx / total_distributed) * 0.7; // [0.15..0.85]
            let wp_count = world.waypoints_world.len().max(2);
            let target_wp = ((fraction * wp_count as f32) as usize).min(wp_count - 2);
            let wp = world.waypoints_world[target_wp];
            let wp_next = world.waypoints_world[target_wp + 1];
            let dir = (wp_next - wp).normalize_or_zero();
            let p = Vec3::new(wp.x, world.start_pos.y, wp.y);
            let angle = dir.y.atan2(dir.x);
            let progress = world.get_track_progress(p.x, p.z);
            (p, Quat::from_rotation_y(-angle), progress)
        };
        let elite_range_end = 1 + race.elite_pool.len();
        let mut spawn_model_idx = i % 5;
        if slot_is_champion {
            if let Some(idx) = race.best_car_model_idx {
                spawn_model_idx = idx as usize;
            }
        } else if i >= 1 && i < elite_range_end {
            if i - 1 < race.elite_pool.len() {
                let elite = &race.elite_pool[i - 1];
                spawn_model_idx = elite.car_model_idx as usize;
            }
        }
        let (racer_entity, car_model) = match spawn_model_idx % 5 {
            0 => (
                crate::prefabs::spawn_porsche_911_gt3(
                    commands,
                    Transform::from_translation(spawn_pos).with_rotation(spawn_rot),
                    &mut *meshes,
                    racer_mat.clone(),
                ),
                car_models[0],
            ),
            1 => (
                crate::prefabs::spawn_ferrari_296_gt3(
                    commands,
                    Transform::from_translation(spawn_pos).with_rotation(spawn_rot),
                    &mut *meshes,
                    racer_mat.clone(),
                ),
                car_models[1],
            ),
            2 => (
                crate::prefabs::spawn_mclaren_720s_gt3(
                    commands,
                    Transform::from_translation(spawn_pos).with_rotation(spawn_rot),
                    &mut *meshes,
                    racer_mat.clone(),
                ),
                car_models[2],
            ),
            3 => (
                crate::prefabs::spawn_aston_vantage_gt3(
                    commands,
                    Transform::from_translation(spawn_pos).with_rotation(spawn_rot),
                    &mut *meshes,
                    racer_mat.clone(),
                ),
                car_models[3],
            ),
            _ => (
                crate::prefabs::spawn_audi_r8_lms_gt3(
                    commands,
                    Transform::from_translation(spawn_pos).with_rotation(spawn_rot),
                    &mut *meshes,
                    racer_mat.clone(),
                ),
                car_models[4],
            ),
        };

        commands
            .entity(racer_entity)
            .insert(Racer {
                name: if i == 0 && current_best_w.is_some() {
                    "👑 CHAMPION".to_string()
                } else {
                    format!("{}-{}", names[i % names.len()], rng.random_range(10..99))
                },
                car_model: car_model.to_string(),
                car_model_idx: (spawn_model_idx % 5) as u8,
                engine,
                is_dead: false,
                distance: 0.0,
                phase: 0.0,
                amplitude: 0.0,
                pain_memories: Vec::new(),
                stagnation: 0.0,
                prev_pos: spawn_pos,
                fuel: crate::simulation::racer::FUEL_INICIAL,
                hp: crate::simulation::racer::HP_INICIAL,
                total_laps: 0,
                memory: [0.0; 5],
                visual_cortex: [0; RETINA_TOTAL],
                tire_temp: 0.0,
                engine_power: 1.0,
                grass_time: 0.0,
                wall_time: 0.0,
                aquaplaning_timer: 0.0,
                joy_integral: 0.0,
                joy_streak: 0.0,
                wall_hits: 0,
                wall_cooldown: 0.0,
                last_waypoint: 0,
                track_progress: spawn_progress,
                grace_ticks: 0,
                reverse_timer: 0.0,
                progress_checkpoint: spawn_progress,
                progress_stall: 0.0,
                wall_ema_l: 15.0,
                wall_ema_r: 15.0,
                wall_dist_prev_l: 15.0,
                wall_dist_prev_r: 15.0,
                wall_dist_prev_f: 50.0,
                proximity_sustain: 0.0,
                post_reverse_lockout: 0.0,
                spawn_progress,
                path: vec![spawn_pos],
                path_timer: 0.0,
                brain_trace: Vec::with_capacity(15),
                trace_timer: 0.0,
                gaze_yaw: 0.0,
                lap_dist_m: 0.0,
                max_progress: spawn_progress,
                slow_time: 0.0,
                race_time: 0.0,
                prev_lap_prog: spawn_progress.rem_euclid(world.lap_distance.max(1.0)),
                post_lap_grace: 0,
                post_lap_checkpoint: 0.0,
                death_reason: None,
                death_frame: 0,
                death_pos: Vec3::ZERO,
                peak_speed: 0.0,
                telemetry: crate::simulation::racer::RacerTelemetry::new(),
                brain_tick_counter: 0,
                last_action: (0.0, 0.0),
                // Tier 4.3: alternates niche by index. 50% defensive / 50% aggressive.
                niche: (i % 2) as u8,
                // Tier 5.1 LITE: data layer — zeroed at spawn; populated via
                // EnvironmentalState::closest_racers in a follow-up (not wired).
                perceived_rivals: [0.0; 6],
                // Tier 6.3 LITE: morpho genes inherited/mutated according to the reproduction branch.
                morpho: morpho_this.sanitized(),
                // v13 KAN output head: 5% injection (v12 10% → HOLE=5181 regression).
                // Untrained fresh heads fell off the track. Mut rate 0.30→0.15.
                kan_out: {
                    let mut head = crate::kan::KanOutputHead::new();
                    // Fallback (empty pool, CHAMPION.dna loaded): slot 0 uses the DNA's knots;
                    // other slots inherit + moderate mutation for diversity.
                    if race.elite_pool.is_empty() {
                        if current_best_w.is_some() {
                            head.knots = current_best_kan_knots;
                            if i >= 3 {
                                use rand::prelude::*;
                                let mut local_rng = rand::rng();
                                head.mutate_spline_knots(|| local_rng.random::<f32>(), 0.15);
                            }
                        }
                    } else {
                        use rand::prelude::*;
                        let mut local_rng = rand::rng();
                        let pool_sz = race.elite_pool.len();
                        if i == 0 && current_best_w.is_some() {
                            head.knots = race.elite_pool[0].kan_knots;
                        } else if local_rng.random::<f32>() < 0.05 {
                            // DIVERSITY INJECTION: 5% fresh head + moderate mut (v13).
                            head.mutate_spline_knots(|| local_rng.random::<f32>(), 0.15);
                        } else {
                            // Blend 2 random parents + light mut.
                            let pa = local_rng.random_range(0..pool_sz);
                            let pb = local_rng.random_range(0..pool_sz);
                            let ka = &race.elite_pool[pa].kan_knots;
                            let kb = &race.elite_pool[pb].kan_knots;
                            for o in 0..3 {
                                for k in 0..crate::kan::KAN_OUT_KNOTS {
                                    head.knots[o][k] = if local_rng.random::<f32>() < 0.5 {
                                        ka[o][k]
                                    } else {
                                        kb[o][k]
                                    };
                                }
                            }
                            head.mutate_spline_knots(|| local_rng.random::<f32>(), 0.04);
                        }
                    }
                    head
                },
                stable_seed: if i == 0 { 42 } else { rng.random::<u64>() },
                sensory_snapshot: [0i8; 64], // Vision data captured each frame for UI
            })
            .with_children(|p| {
                p.spawn((
                    Mesh3d(wing.clone()),
                    MeshMaterial3d(racer_mat.clone()),
                    Transform::from_xyz(-1.2, 0.4, 0.0),
                ));
                p.spawn((
                    Mesh3d(cockpit.clone()),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        emissive: LinearRgba::WHITE * 2.0,
                        ..default()
                    })),
                    Transform::from_xyz(0.5, 0.3, 0.0),
                ));

                // VISUAL FIELD HEADLIGHT (135-degree FOV, 40-meter range same as the KAN Retina)
                p.spawn((
                    SpotLight {
                        color: Color::WHITE,
                        intensity: 150_000.0, // High beam for extended range (150m)
                        range: 150.0,
                        radius: 0.0,
                        shadows_enabled: false,
                        inner_angle: 0.8,
                        outer_angle: 1.178, // (135 degrees) / 2 in radians
                        ..default()
                    },
                    Transform::from_xyz(1.5, 0.5, 0.0) // At the car's nose
                        .with_rotation(
                            Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2)
                                .mul_quat(Quat::from_rotation_x(-0.05)),
                        ), // Turns forward (-Z -> +X)
                ));
            });
    }

    // Tier 4.4: protected-zone log (1x per gen, on the first spawn batch).
    if idx_start == 0 {
        if let Some(mut file) = open_report_log() {
            let _ = writeln!(
                file,
                "[PROTECTED] gen {} protected zone: {} weights exempt",
                race.generation, PROTECTED_ZONE_SIZE
            );
        }
    }

    // Tier 4.2: log of this generation's mean epigenetic inheritance (helps visualize
    // the evolution of "traumatized" vs "calm" vs "explorer" lineages).
    if epi_inherited_count > 0 {
        if let Some(mut file) = open_report_log() {
            let n = epi_inherited_count as f32;
            let _ = writeln!(
                file,
                "[EPIG] gen {} inheritance n={} parent_mean cort={:.3} dopa={:.3} curi={:.3} (child baseline = 10% of these)",
                race.generation,
                epi_inherited_count,
                epi_cortisol_sum / n,
                epi_dopamine_sum / n,
                epi_curiosity_sum / n
            );
        }
    }
}
