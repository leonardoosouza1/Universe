//! Screen — replays a recorded sensory wave through the real brain kernel to score
//! candidate genomes open-loop, in milliseconds instead of the ~18 s a closed-loop
//! physics evaluation costs. Task 4 built the recording; this binary is the payoff.
//!
//! Reuses `gpu::brain::BrainGpu` end to end — no perception, no physics, and NO CPU
//! reimplementation of KAN/CfC/BitLinear/periods/thresholds/activations. This
//! repository has already shipped two disjoint brains once (`cortex_evolve` ended up
//! with zero references to `SovereignEngine`/`SlowBrain`); a CPU replica of the brain
//! kernel would reintroduce exactly that failure with no test able to see it.
//!
//! ```text
//! ./target/release/screen --wave <wave.f32> --genomes <dir of .bin> \
//!     [--ticks N] [--threshold 0.26] [--out-beta 0.85]
//! ```

use universe::gpu::brain::BrainGpu;
use universe::gpu::genome_io::load_genome;
use universe::gpu::wave_record::{self, WAVE_CHANNELS};

/// Tick length `gpu_evolve` runs at (64 Hz). The recordings this screen replays were
/// made at this rate, and the brain kernel's CfC integration is `dt`-dependent.
const DT: f32 = 1.0 / 64.0;

/// Mean absolute error per tick across the three commands — the quantity
/// `docs/DESTILABILIDADE.md` calibrated: 0.26 compounds in closed loop and the clone
/// stalls, 0.18 does not and it drives.
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
    let mut threshold = 0.26f32;
    let mut out_beta = universe::gpu::brain::OUT_BETA_DEFAULT;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--wave" => {
                i += 1;
                wave_path = args[i].clone();
            }
            "--genomes" => {
                i += 1;
                genomes_dir = args[i].clone();
            }
            "--ticks" => {
                i += 1;
                ticks_max = args[i].parse().unwrap_or(usize::MAX);
            }
            "--threshold" => {
                i += 1;
                threshold = args[i].parse().unwrap_or(0.26);
            }
            "--out-beta" => {
                i += 1;
                // Per-run CLI override in `gpu_evolve` (`--fadiga-saida`), not stored in
                // the genome — the wave file carries no header of the source run's
                // brain parameters. Default is `OUT_BETA_DEFAULT`, today's production
                // value; pass this when screening against a recording made under a
                // different override.
                out_beta = args[i].parse().unwrap_or(out_beta);
            }
            _ => {}
        }
        i += 1;
    }

    let (waves, real) = wave_record::read_all(&wave_path)
        .unwrap_or_else(|e| panic!("[ERROR] recording {wave_path}: {e}"));

    // Skip row 0: `gpu_evolve/main.rs:2799` runs a JIT warm-up tick whose return value
    // is discarded but whose side effects (the CfC state it advances) are not, so a
    // 20 000-tick run produces 20 001 rows. Row 0 is that warm-up tick, carrying CfC
    // state this screen never reproduces (every genome starts at the true zero initial
    // state) — scoring it would compare a tick this screen structurally cannot match.
    assert!(
        real.len() >= 2,
        "[ERROR] {wave_path}: {} row(s), need at least 2 (row 0 is the warm-up tick)",
        real.len()
    );
    let waves = &waves[WAVE_CHANNELS..];
    let real = &real[1..];

    let n_ticks = real.len().min(ticks_max);
    println!(
        "recording {n_ticks} ticks (row 0 skipped as warm-up) \
         · threshold {threshold:.3} · out_beta {out_beta:.2}"
    );

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&genomes_dir)
        .unwrap_or_else(|e| panic!("[ERROR] {genomes_dir}: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    files.sort();
    println!("{} candidate(s)", files.len());

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some(genome) = load_genome(path.to_str().unwrap()) else {
            println!("  {name:<34} [ERROR] could not load");
            continue;
        };

        let t0 = std::time::Instant::now();
        let mut brain = match BrainGpu::new_for_genome(&genome) {
            Ok(b) => b,
            Err(e) => {
                println!("  {name:<34} [ERROR] {e}");
                continue;
            }
        };
        brain.out_beta = out_beta;

        let mut pred: Vec<[f32; 3]> = Vec::with_capacity(n_ticks);
        for t in 0..n_ticks {
            let wave = &waves[t * WAVE_CHANNELS..(t + 1) * WAVE_CHANNELS];
            pred.push(brain.step_from_wave(wave, DT));
        }
        let elapsed = t0.elapsed();

        let e = error_per_tick(&pred, &real[..n_ticks]);
        let verdict = if e > threshold { "REJECT" } else { "keep" };
        println!(
            "  {name:<34} error/tick {e:.4}  {verdict}  ({:.1} ms/genome, {:.0} ticks/s)",
            elapsed.as_secs_f64() * 1000.0,
            n_ticks as f64 / elapsed.as_secs_f64()
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
