//! Sensitivity gate — can each gene block still move the steering wheel? NO GPU.
//!
//! Port of `neuromod-lab/src/testes_sentidos.py:1-3` asked for by C3 of
//! `docs/PRD-CEREBRO-NEUROMOD.md`. It perturbs ONE gene block at a time by ±δ, replays a
//! recorded sensory wave open-loop through the CPU brain, and prints a group × output
//! matrix of how much each block moved `throttle`, `steer` and `brake`. In the lab the
//! same check took 30 s and would have caught finding 84 — sixteen body senses invisible
//! to the steering.
//!
//! ```text
//! ./target/release/sensibilidade --genoma <champion.bin> --onda <record.uwav>
//! ./target/release/sensibilidade --genoma <champion.bin> --onda-sintetica 64
//! ```
//!
//! Exit codes: `0` the gate passed, `1` the gate failed (at least one group is blind, or
//! an off-path group moved something), `2` bad usage or unreadable input.
//!
//! # No GPU, on purpose
//!
//! The gate has to be runnable BEFORE anything enters the shared GPU queue (PRD §0.4),
//! so it replays through `SovereignEngine` (`src/engine.rs:195-260`), the CPU brain the
//! repository already has, and never opens a `wgpu` device. It does not reimplement
//! `brain.wgsl`: `src/bin/screen.rs`'s header records that this project already shipped
//! two disjoint brains once, and a CPU replica written for this gate would be the third.
//!
//! # Why a group can be legitimately blind here (read this before filing a bug)
//!
//! `SovereignEngine` is not `BrainGpu`, and a `UWAV` recording carries 192 sensory
//! channels and nothing else. Three classes of block therefore CANNOT move anything in
//! this replay, and `gpu::sensitivity::groups()` declares each one with its evidence:
//!
//! 1. **the block lives in the GPU kernel** — the four P1.7 mechanisms
//!    (`neuron_alpha`, `neuron_period_l1`, `neuron_threshold_l1`, `neuron_activation`)
//!    exist only in `brain.wgsl`;
//! 2. **the block lives in the host loop of a binary** — the output KAN, the slow
//!    brain's modulations and the M4 conductor are applied in
//!    `gpu_evolve/context.rs:1600-1624` and `:1862-1907`, binary code this library
//!    cannot call. `slow` is the sharpest case: `apply_to_engine` really does load the
//!    1 091 weights into the engine, and `execute_cycle` never ticks it;
//! 3. **the block reads state the recording does not carry** — `sensors` drives
//!    `perception.wgsl`, and an open-loop replay consumes an already-recorded wave, so
//!    no perception step runs.
//!
//! Those rows print as `off-path`, with the reason, and do NOT fail the gate. The
//! declaration is still falsifiable: an off-path group that moves an output prints
//! `UNEXPECTED SIGNAL` and fails, because then the list is wrong.
//!
//! # `--mem-sintetico`: documented, not implemented, and that is deliberate
//!
//! Class 3 above is the only one a synthetic input can repair, and it is the class the
//! hippocampus C1 adds will land in: its input is a per-sector memory written by lap
//! events, and no wave row contains one, so a faithful replay would feed it zeros and
//! the block would read blind for a reason that has nothing to do with wiring. The fix
//! is to inject a FIXED non-zero memory — the `--mem-sintetico` flag the PRD asks for.
//! `Genome` has no `hip` block on this branch (`src/gpu/evolution.rs:93-182`), so there
//! is nothing to inject into, and the flag is not accepted: a flag that silently does
//! nothing is worse than a documented absence. C1 adds the block, the group, and this
//! flag together.
// check-language: allow -- the two gate lines below and this binary's name are the
// machine contract fixed by docs/PRD-CEREBRO-NEUROMOD.md §C3; a workflow greps them
// verbatim. Everything else in this file is English, per docs/STYLE.md §1.

use universe::gpu::dof::{DEFAULT_GENE_THRESHOLD, dof_by_block};
use universe::gpu::genome_io::load_genome;
use universe::gpu::sensitivity::{DEFAULT_DELTAS, run_gate, synthetic_wave};
use universe::gpu::wave_record;

/// Tick length used with `--onda-sintetica`, when no recording header supplies one.
/// 1/64 s is what `gpu_evolve` runs at today; a real recording's own `WaveConfig::dt`
/// always wins over this (guessing `dt` is the class of bug the `UWAV` header exists to
/// remove — see `gpu/wave_record.rs`'s module doc).
const DEFAULT_DT: f32 = 1.0 / 64.0;

fn usage() -> ! {
    eprintln!(
        "usage: sensibilidade --genoma <genome.bin> \
         (--onda <record.uwav> | --onda-sintetica <ticks>) [--dt <seconds>] [--delta <d>]...\n\
         \n\
         --genoma           UGEN v1/v2 genome, read with the canonical parser \
         (gpu::genome_io::load_genome)\n\
         --onda             UWAV recording; its header supplies dt\n\
         --onda-sintetica   closed-form fixture wave of N ticks, for a wiring check when \
         no recording is at hand\n\
         --dt               overrides the tick length (seconds); default {DEFAULT_DT}\n\
         --delta            perturbation size, repeatable; default {DEFAULT_DELTAS:?}"
    );
    std::process::exit(2);
}

fn next_value(args: &[String], i: usize, flag: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| {
        eprintln!("[ERROR] {flag} requires a value");
        std::process::exit(2);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut genome_path: Option<String> = None;
    let mut wave_path: Option<String> = None;
    let mut synthetic_ticks: Option<usize> = None;
    let mut dt_override: Option<f32> = None;
    let mut deltas: Vec<f32> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].clone();
        i += 1;
        match flag.as_str() {
            "--genoma" => {
                genome_path = Some(next_value(&args, i, "--genoma"));
                i += 1;
            }
            "--onda" => {
                wave_path = Some(next_value(&args, i, "--onda"));
                i += 1;
            }
            "--onda-sintetica" => {
                let v = next_value(&args, i, "--onda-sintetica");
                i += 1;
                synthetic_ticks = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("[ERROR] --onda-sintetica expects a tick count, got '{v}'");
                    std::process::exit(2);
                }));
            }
            "--dt" => {
                let v = next_value(&args, i, "--dt");
                i += 1;
                dt_override = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("[ERROR] --dt expects seconds, got '{v}'");
                    std::process::exit(2);
                }));
            }
            "--delta" => {
                let v = next_value(&args, i, "--delta");
                i += 1;
                deltas.push(v.parse().unwrap_or_else(|_| {
                    eprintln!("[ERROR] --delta expects a number, got '{v}'");
                    std::process::exit(2);
                }));
            }
            "--help" | "-h" => usage(),
            other => {
                eprintln!("[ERROR] unknown flag '{other}'");
                usage();
            }
        }
    }

    let Some(genome_path) = genome_path else {
        eprintln!("[ERROR] --genoma is required");
        usage();
    };
    let Some(genome) = load_genome(&genome_path) else {
        eprintln!("[ERROR] {genome_path}: not a readable UGEN genome");
        std::process::exit(2);
    };

    // A recording and a fixture are mutually exclusive: silently preferring one would
    // leave the reader unsure which wave the printed matrix was measured against.
    let (waves, dt, source) = match (&wave_path, synthetic_ticks) {
        (Some(_), Some(_)) => {
            eprintln!("[ERROR] --onda and --onda-sintetica are mutually exclusive");
            std::process::exit(2);
        }
        (Some(p), None) => match wave_record::read_all(p) {
            Ok((w, _intents, cfg)) => {
                let dt = dt_override.unwrap_or(cfg.dt);
                (w, dt, format!("recording {p}"))
            }
            Err(e) => {
                eprintln!("[ERROR] {e}");
                std::process::exit(2);
            }
        },
        (None, Some(t)) => (
            synthetic_wave(t),
            dt_override.unwrap_or(DEFAULT_DT),
            format!("synthetic fixture, {t} ticks"),
        ),
        (None, None) => {
            eprintln!("[ERROR] one of --onda / --onda-sintetica is required");
            usage();
        }
    };

    if deltas.is_empty() {
        deltas.extend_from_slice(&DEFAULT_DELTAS);
    }

    println!("genome  {genome_path}");
    println!("wave    {source}  (dt = {dt} s)");
    let dofs = dof_by_block(&genome, DEFAULT_GENE_THRESHOLD);
    let dof_line: Vec<String> = dofs.iter().map(|(name, n)| format!("{name}={n}")).collect();
    println!(
        "dof     |w| > {DEFAULT_GENE_THRESHOLD}: {}",
        dof_line.join("  ")
    );
    println!();

    let report = match run_gate(&genome, &waves, dt, &deltas) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[ERROR] {e}");
            std::process::exit(2);
        }
    };
    print!("{report}");
    println!();

    let failures = report.failures();
    if failures.is_empty() {
        println!("SENSIBILIDADE ok");
    } else {
        let names: Vec<&str> = failures.iter().map(|r| r.name).collect();
        println!("SENSIBILIDADE FALHOU: {}", names.join(", "));
        std::process::exit(1);
    }
}
