//! Screen — replays a recorded sensory wave through the real brain kernel (fast brain
//! + M4 region gates) to score candidate genomes open-loop, batched across many
//! candidates per GPU dispatch.
//!
//! Reuses `gpu::brain::BrainGpu`, `gpu::slow_brain::SlowBrainGpu` and
//! `gpu::screen_brain::ScreenBrain` end to end — no perception, no physics, and NO CPU
//! reimplementation of KAN/CfC/BitLinear/periods/thresholds/activations/the M4
//! conductor. This repository has already shipped two disjoint brains once
//! (`cortex_evolve` ended up with zero references to `SovereignEngine`/`SlowBrain`); a
//! CPU replica of any of these layers would reintroduce exactly that failure with no
//! test able to see it.
//!
//! ```text
//! ./target/release/screen --wave <wave.f32> --genomes <dir of .bin> \
//!     [--ticks N] [--threshold 0.60] [--batch 48]
//! ```
//!
//! **On `--threshold`, read `docs/SCREEN_CALIBRATION.md` §6 before changing it.** This
//! binary was originally built as a rejection funnel at 0.26 (`docs/DESTILABILIDADE.md`'s
//! figure for a *distilled pilot replacing the brain*, never for genomes of this
//! architecture). Calibrated against 85 candidates and closed-loop ground truth, that
//! transplant was refuted: 6 of the 7 candidates it rejects drive, 2 of them better than
//! the recording champion, and the score does not rank by quality (Spearman −0.05). The
//! default here is **0.60**, re-scoped from a pruning threshold to a tripwire for
//! catastrophic divergence — a destroyed brain scored 0.9066 against ≤0.4287 for every
//! candidate that still drove. That gap is real but **0.60 is n = 1 on each side**, a
//! hypothesis to be calibrated with more reference brains, not a validated cutoff.
//!
//! All ten runtime flags that change the brain's forward pass (`out_beta`, `out_gain`,
//! `norm_l2_by_std`, sensor noise/dropout, the four P1.7 mechanism switches,
//! `--neuromod-regions`), plus `dt`, `--brain-skip` and whether a ghost pilot was
//! active, are read from the recording's own `wave_record::WaveConfig` header — there
//! is no `--out-beta`-style flag on this CLI to guess wrong. `read_all` refuses a
//! legacy (headerless), noisy, or pilot-driven recording outright rather than
//! defaulting: see `wave_record.rs`'s module doc for why.

use universe::gpu::evolution::Genome;
use universe::gpu::genome_io::load_genome;
use universe::gpu::screen_brain::ScreenBrain;
use universe::gpu::wave_record::{self, WAVE_CHANNELS};

/// Mean absolute error per tick across the three commands. This is NOT the
/// `docs/DESTILABILIDADE.md` 0.26/0.18 pair — that pair was measured for a distilled
/// pilot replacing the brain and was refuted as a threshold for genomes of this
/// architecture in `docs/SCREEN_CALIBRATION.md` §6 (6 of 7 rejects at 0.26 drive, 2
/// better than the champion). The score is real and bit-exact against its own
/// recording; only its use as a pruning cutoff was refuted.
fn error_per_tick(pred: &[[f32; 3]], real: &[[f32; 3]]) -> f32 {
    let n = pred.len().min(real.len());
    if n == 0 {
        return f32::INFINITY;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        for k in 0..3 {
            sum += (pred[i][k] - real[i][k]).abs() as f64;
        }
    }
    (sum / (n as f64 * 3.0)) as f32
}

/// Reads the value following `flag` at `args[i]`, or exits with a usage error. A
/// trailing flag with no value (`screen --wave`) used to index out of bounds and
/// panic with no explanation — this names the flag that was left dangling.
fn next_value(args: &[String], i: usize, flag: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| {
        eprintln!("[ERROR] {flag} requires a value");
        std::process::exit(2);
    })
}

fn main() {
    // SIGSEGV fix (paid for by physics/perception/brain already): multiple Vulkan ICDs
    // cause a segfault when the loader picks one incompatible with the AMD GPU. Must be
    // set before the FIRST `wgpu::Instance::new` — the loader loads ICDs once, globally.
    // Same guard as `gpu_evolve/main.rs:540-546`.
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
        // safety: single thread, no wgpu instance created yet
        unsafe {
            std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let mut wave_path = String::new();
    let mut genomes_dir = String::new();
    let mut ticks_max = usize::MAX;
    let mut threshold = 0.60f32;
    let mut batch_size = 48usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--wave" => {
                i += 1;
                wave_path = next_value(&args, i, "--wave");
            }
            "--genomes" => {
                i += 1;
                genomes_dir = next_value(&args, i, "--genomes");
            }
            "--ticks" => {
                i += 1;
                ticks_max = next_value(&args, i, "--ticks")
                    .parse()
                    .unwrap_or(usize::MAX);
            }
            "--threshold" => {
                i += 1;
                threshold = next_value(&args, i, "--threshold").parse().unwrap_or(0.60);
            }
            "--batch" => {
                i += 1;
                batch_size = next_value(&args, i, "--batch").parse().unwrap_or(48).max(1);
            }
            _ => {}
        }
        i += 1;
    }

    let (waves, real, config) = wave_record::read_all(&wave_path)
        .unwrap_or_else(|e| panic!("[ERROR] recording {wave_path}: {e}"));

    // Row 0 is a JIT warm-up tick (`gpu_evolve/main.rs:2857`): its return value is
    // discarded but its side effects are not, so a 20 000-tick run produces 20 001
    // rows, and by the time row 1 (the first SCORED tick) happens, the true recording
    // has already advanced its CfC state, its P1.7 period counter, and (with
    // `--neuromod-regions`) its M4 reserve by one dispatch.
    //
    // The first cut of this screen skipped row 0 outright, reasoning that "the CfC
    // state it left behind is not reproducible anyway". That reasoning was WRONG in a
    // way that mattered a lot more than expected: the P1.7 period mechanism gates each
    // L1 neuron's recompute on a per-neuron COUNTER that only advances once per brain
    // dispatch. Skipping row 0 puts every genome's counter permanently one dispatch
    // BEHIND the recording's — not a decaying transient, but a standing phase offset
    // for the rest of the run, since the counter never resynchronises on its own. That
    // was the real cause of the review round's "flat, non-decaying 0.04-0.05 floor"
    // finding (C1) — M4 gates were a real, smaller contributor, not the dominant one.
    //
    // The fix: REPLAY row 0 through the exact same kernel (fast brain + M4 gate
    // machinery), from the true zero initial state — that is what production ALSO
    // does; a fresh being's CfC/reserve state was never anything but zero/neutral —
    // and discard only its OWN prediction, never comparing it against `real[0]`. Row 1
    // onward then lines up counter-for-counter with the recording, and this warm-up
    // replay is the reason the parity check below reproduces `real` bit-for-bit.
    assert!(
        real.len() >= 2,
        "[ERROR] {wave_path}: {} row(s), need at least 2 (row 0 is the warm-up tick)",
        real.len()
    );
    let warmup_wave = waves[0..WAVE_CHANNELS].to_vec();
    let waves = &waves[WAVE_CHANNELS..];
    let real = &real[1..];

    let n_ticks_full = real.len();
    let n_ticks = n_ticks_full.min(ticks_max);
    let partial = n_ticks < n_ticks_full;

    if config.brain_skip != 1 {
        println!(
            "[WARN] recording made with --brain-skip {} — replay uses this recording's own \
             `dt` ({:.5} s) per row, which matches what the brain kernel actually integrated \
             over per firing regardless of brain_skip (see WaveConfig::dt's doc); untested \
             beyond that.",
            config.brain_skip, config.dt
        );
    }
    println!(
        "recording {n_ticks}/{n_ticks_full} ticks (row 0 replayed as warm-up, not scored) \
         · threshold {threshold:.3} · dt {:.5} · out_beta {:.2} · neuromod_regioes {}",
        config.dt,
        config.out_beta,
        config.neuromod_regioes != 0
    );
    if partial {
        println!(
            "[WARN] --ticks {ticks_max} scores a PARTIAL window ({n_ticks} of \
             {n_ticks_full}) — verdicts are suppressed because a partial window is known \
             to flip them: against this recording's own gen0000 checkpoint, 2000 ticks \
             scores 0.2790 (would REJECT at threshold 0.26) while the full 20000 scores \
             0.2240 (keep) — measured on this exact recording, not a hypothetical."
        );
    }

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&genomes_dir)
        .unwrap_or_else(|e| panic!("[ERROR] {genomes_dir}: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    println!("{} candidate(s), batch size {batch_size}", files.len());

    for chunk in files.chunks(batch_size) {
        let mut names: Vec<String> = Vec::with_capacity(chunk.len());
        let mut genomes: Vec<Genome> = Vec::with_capacity(chunk.len());
        for path in chunk {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            match load_genome(path.to_str().unwrap()) {
                Some(g) => {
                    names.push(name);
                    genomes.push(g);
                }
                None => println!("  {name:<34} [ERROR] could not load"),
            }
        }
        if genomes.is_empty() {
            continue;
        }

        let t0 = std::time::Instant::now();
        let mut screen = match ScreenBrain::new_for_genomes(&genomes, config.neuromod_regioes != 0)
        {
            Ok(s) => s,
            Err(e) => {
                for name in &names {
                    println!("  {name:<34} [ERROR] {e}");
                }
                continue;
            }
        };
        // `new_for_genomes` leaves the brain at `gpu_evolve`'s DEFAULTS
        // (out_beta/out_gain=OUT_*_DEFAULT, norm_l2_by_std=true, all P1.7 mechanisms
        // on) — the recording's own header is what says whether THIS run actually
        // matched those defaults. Wiring it here is exactly the step the first cut of
        // this screen skipped, which is how it silently scored 0.4772 against a
        // recording made under out_beta=0.0 while reporting out_beta as if it were
        // 0.85 nowhere at all.
        screen.brain.out_beta = config.out_beta;
        screen.brain.out_gain = config.out_gain;
        screen.brain.norm_l2_by_std = config.norm_l2_by_std != 0;
        screen.brain.mem_off = config.mem_off != 0;
        screen.brain.period_off = config.period_off != 0;
        screen.brain.threshold_off = config.threshold_off != 0;
        screen.brain.activation_off = config.activation_off != 0;

        // Warm-up: row 0's wave, replayed once to advance CfC state, the P1.7 period
        // counter and the M4 reserve from the true zero/neutral initial state —
        // exactly what the recording's own row 1 was scored against. Its own
        // prediction is discarded; see the long comment above `warmup_wave` for why
        // this (not skipping row 0 outright) is what makes row 1 onward line up.
        let _ = screen.step(&warmup_wave, config.dt);

        let mut preds: Vec<Vec<[f32; 3]>> = vec![Vec::with_capacity(n_ticks); genomes.len()];
        for t in 0..n_ticks {
            let wave = &waves[t * WAVE_CHANNELS..(t + 1) * WAVE_CHANNELS];
            let step = screen.step(wave, config.dt);
            for (g, cmd) in step.into_iter().enumerate() {
                preds[g].push(cmd);
            }
        }
        let elapsed = t0.elapsed();
        let per_genome_ms = elapsed.as_secs_f64() * 1000.0 / genomes.len() as f64;
        let ticks_per_s = (n_ticks * genomes.len()) as f64 / elapsed.as_secs_f64();

        for (g, name) in names.iter().enumerate() {
            let e = error_per_tick(&preds[g], &real[..n_ticks]);
            if partial {
                println!("  {name:<34} error/tick {e:.4}  (partial window, no verdict)");
            } else {
                let verdict = if e > threshold { "DIVERGED" } else { "keep" };
                println!("  {name:<34} error/tick {e:.4}  {verdict}");
            }
        }
        println!(
            "  [batch of {}] {:.1} ms/genome, {:.0} ticks/s aggregate",
            genomes.len(),
            per_genome_ms,
            ticks_per_s
        );
    }
}

#[cfg(test)]
mod tests {
    use super::error_per_tick;

    #[test]
    fn a_perfect_clone_scores_zero() {
        let a = vec![[0.5, -0.2, 0.0], [0.1, 0.1, 0.1]];
        assert!(error_per_tick(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn the_error_is_the_mean_over_ticks_and_channels() {
        let pred = vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let real = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        // one channel off by 1.0, out of 2 ticks x 3 channels = 6 slots
        assert!((error_per_tick(&pred, &real) - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn mismatched_lengths_score_over_the_common_prefix() {
        let pred = vec![[1.0, 0.0, 0.0]];
        let real = vec![[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]];
        assert!((error_per_tick(&pred, &real) - 1.0 / 3.0).abs() < 1e-6);
    }
}
