//! Open-loop CPU replay, and the closed-form wave that stands in for a recording.
//!
//! Split out of `sensitivity.rs` proper when it crossed `docs/STYLE.md`'s 600-line cap:
//! the parent module is about GROUPS, perturbation and the verdict; these two functions
//! are about running the brain once. Same split, and the same reason, as
//! `gpu/brain/screen.rs` out of `gpu/brain`.
//!
//! Everything here is CPU-only and deterministic. There is deliberately NO CPU replica of
//! `brain.wgsl` — see the parent module's header for why that prohibition is load-bearing
//! in this repository.

use crate::config::iara::brain::SENSORY_WAVE_SIZE;
use crate::config::iara::vision::RETINA_TOTAL;
use crate::engine::SovereignEngine;
use crate::gpu::evolution::Genome;

/// Replays a recorded wave through the CPU brain and returns `[throttle, steer, brake]`
/// per tick — `final_intent[0..3)`, the same three slots `gpu_evolve/context.rs` writes
/// into a `UWAV` row (see `gpu/wave_record.rs`), so a trace from here is comparable with
/// the commands stored in the recording.
///
/// `waves` is `ticks × SENSORY_WAVE_SIZE` f32, row-major — exactly what
/// `wave_record::read_all` returns. Each value is an `i8` channel widened to f32, so
/// narrowing it back is exact rather than a rounding step.
///
/// # What this wires beyond `apply_to_engine`
///
/// `Genome::apply_to_engine` copies L1, L2 and the slow brain and stops
/// (`evolution.rs:680-701`). Two things are set here on top of it, both documented:
///
/// - **`tau` into the CfC cell.** `Genome::tau` IS the per-channel decay rate in s⁻¹
///   (`evolution.rs:111-117`) and `CfCCell::decay_rates` is where that lives
///   (`cfc.rs:56-68`); `apply_to_engine` simply never connected them, which is why the
///   τ bank would otherwise read as off-path. Skipped, not truncated, when the genome's
///   τ length does not match the cell's — a short vector would leave half the channels
///   at the old τ, the mute half-heterogeneous bug `set_decay_rates` asserts against.
/// - **A zero champion snapshot.** The mirror-neuron block (`engine.rs:588-606`) falls
///   back to a PROCESS-GLOBAL champion retina when the per-engine one is `None`. Left
///   unset, any other test in the process that publishes a champion would change this
///   replay's output. A zeroed snapshot makes the cosine exactly 0, so the block never
///   fires and the replay depends on nothing outside this function.
///
/// `target_phase` is fixed at 0.0: the heart resonator is a closed-loop signal a replay
/// has no source for, and holding it constant keeps the two arms of every comparison on
/// the same footing.
///
/// # This brain is never bit-silent
///
/// Measured while building the gate: feeding an all-zero wave does NOT produce a
/// zero command. `CfCCell` injects deterministic denoising noise into its state every
/// tick (`cfc.rs:79-81`, `denoise_mag` = 0.10), so `filtered_retina` is non-zero with no
/// input at all and L1 still has something to weigh. It is deterministic — the noise
/// comes from a counter hash, not an RNG — so two replays of the same genome are still
/// bit-identical; but a "silent wave" is not a valid null control here. Use a zero
/// PERTURBATION for that.
pub fn replay_open_loop(g: &Genome, waves: &[f32], dt: f32) -> Vec<[f32; 3]> {
    let ticks = waves.len() / SENSORY_WAVE_SIZE;
    let mut eng = SovereignEngine::new();
    g.apply_to_engine(&mut eng);
    if g.tau.len() == eng.cfc_state.hidden_size {
        eng.cfc_state.set_decay_rates(&g.tau);
    }
    eng.champion_retina_snapshot = Some([0.0; RETINA_TOTAL]);
    eng.champion_action = Some([0.0, 0.0]);

    let mut out = Vec::with_capacity(ticks);
    let mut row = [0i8; SENSORY_WAVE_SIZE];
    for t in 0..ticks {
        let base = t * SENSORY_WAVE_SIZE;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = waves[base + c] as i8;
        }
        eng.execute_cycle(&row, 0.0, dt);
        out.push([
            eng.final_intent[0],
            eng.final_intent[1],
            eng.final_intent[2],
        ]);
    }
    out
}

/// A deterministic synthetic wave, for when no `UWAV` recording is at hand.
///
/// Closed-form, no RNG: channel `c` is a sinusoid whose frequency and phase are fixed
/// functions of `c`, scaled into the i8 range the real recorder writes. Two properties
/// the gate needs and a constant or random wave would not give:
///
/// - **it varies across ticks**, so the CfC's per-channel τ has something to filter —
///   against a constant input every τ settles to the same DC value and the `tau` block
///   would read as blind for a reason that has nothing to do with wiring;
/// - **channels differ from each other**, so an L1 row cannot be moved by a common mode
///   alone.
///
/// It is a WIRING fixture, never a stand-in for a recording: it proves a block can reach
/// the output, and says nothing about what that block does on a real track.
pub fn synthetic_wave(ticks: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ticks * SENSORY_WAVE_SIZE);
    for t in 0..ticks {
        for c in 0..SENSORY_WAVE_SIZE {
            let freq = 0.03 + (c % 17) as f32 * 0.011;
            let phase = (c as f32) * 0.37;
            let v = 100.0 * (t as f32 * freq + phase).sin();
            // Exactly the recorder's domain: an i8 channel value widened to f32.
            out.push(v as i8 as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{replay_open_loop, synthetic_wave, SENSORY_WAVE_SIZE};
    use crate::gpu::evolution::Genome;

    #[test]
    fn the_synthetic_wave_is_deterministic_and_varies_over_time() {
        let a = synthetic_wave(8);
        let b = synthetic_wave(8);
        assert_eq!(a, b, "same call, same bytes");
        assert_eq!(a.len(), 8 * SENSORY_WAVE_SIZE);
        let first = &a[0..SENSORY_WAVE_SIZE];
        let fourth = &a[3 * SENSORY_WAVE_SIZE..4 * SENSORY_WAVE_SIZE];
        assert_ne!(first, fourth, "a constant wave would mute every tau gene");
    }

    #[test]
    fn the_synthetic_wave_channels_differ_from_each_other() {
        // A wave that carried the same value on all 192 channels would be a common mode:
        // an L1 row could then only be moved by the sum of its weights, and every row of
        // the matrix would understate its block.
        let w = synthetic_wave(1);
        let distinct: std::collections::BTreeSet<i32> = w.iter().map(|v| *v as i32).collect();
        assert!(
            distinct.len() > 20,
            "only {} distinct channel values in one tick",
            distinct.len()
        );
    }

    #[test]
    fn replaying_the_same_genome_twice_gives_the_same_trace() {
        // The whole gate is a difference of two replays; if the replay itself drifted,
        // every row would be noise. This is the guard for the process-global champion
        // snapshot and for any other hidden state.
        let g = Genome::random(0xC3_5E_1751);
        let w = synthetic_wave(24);
        let a = replay_open_loop(&g, &w, 1.0 / 64.0);
        let b = replay_open_loop(&g, &w, 1.0 / 64.0);
        assert_eq!(a, b);
        assert_eq!(a.len(), 24);
    }

    #[test]
    fn two_different_genomes_give_different_traces() {
        // The other half of that guard: a replay that ignored the genome entirely would
        // pass the determinism test above and fill the whole matrix with zeros.
        let w = synthetic_wave(24);
        let a = replay_open_loop(&Genome::random(1), &w, 1.0 / 64.0);
        let b = replay_open_loop(&Genome::random(2), &w, 1.0 / 64.0);
        assert_ne!(a, b);
    }
}
