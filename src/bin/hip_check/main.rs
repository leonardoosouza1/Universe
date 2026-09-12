//! `hip_check` — replays the C1 sector-hippocampus FORMULAS against recorded data. NO GPU.
//!
//! The hippocampus (`src/gpu/hippocampus.rs`, `docs/PRD-CEREBRO-NEUROMOD.md` §C1) is a
//! pure function of two things a run already writes to disk: the 29 `hip` genes at the end
//! of `evolve_best_gen<N>.bin` and the per-tick rows of `hip_gen<N>.f32`. So "is the
//! mechanism doing what the formula says?" never needs a GPU battery — it needs this
//! binary, which recomputes `bias` for every recorded row and compares it with the number
//! the run actually added to the commands.
//!
//! ```text
//! ./target/release/hip_check --genoma <evolve_best_genN.bin> --hip <hip_genN.f32>
//! ./target/release/hip_check --hip <hip_genN.f32> --setores [--genoma <g.bin>]
//! ```
//!
//! Exit codes: `0` the recomputation matched, `1` it did not (or a row is corrupt),
//! `2` bad usage or unreadable input.
//!
//! The unit tests live in `tests.rs` next to this file (the directory-binary layout
//! `src/bin/gpu_evolve/` already uses), so both files stay under the 600-line cap of
//! `docs/STYLE.md` §2. The binary is still `./target/release/hip_check`; Cargo discovers
//! `src/bin/<name>/main.rs` on its own, so no `[[bin]]` entry is needed — the same rule
//! the Cargo.toml comment states for `sensibilidade`.
//!
//! # WHICH genome belongs to `hip_gen<N>.f32` — read this before trusting a mismatch
//!
//! `hip_gen<N>.f32` records **being 0** of generation `N` (`context.rs:2222`), while
//! `evolve_best_gen<N>.bin` saves the being with the **highest fitness** of generation `N`
//! (`main.rs:3630-3637`). Those are not the same being. `evolve()` sorts the previous
//! population by descending fitness and copies the top `n_elite` through UNMUTATED into
//! slots `0..n_elite` (`gpu/evolution.rs:860-873`) — a fact `main.rs:3384` already relies
//! on, where the champion-reproduction block states that `genomes[0]` of this generation
//! is literally the same genome that was the best of the previous one. So **being 0 of
//! generation N carries the genome of
//! `evolve_best_gen<N-1>.bin`**, and equals `evolve_best_gen<N>.bin` only when the elite
//! also won generation `N`. Generation 0 has no previous elite at all: being 0 is random
//! (or whatever `--retomar` loaded), so nothing on disk is guaranteed to be it.
//!
//! Measured on `/media/llm-data/universe-runs/neuromod-lab-taxi/u0c_valida_hip/`
//! (2026-09-12): over the 30 001 rows of `hip_gen0003.f32` the maximum absolute difference
//! against `evolve_best_gen0002.bin` is `0.000e0` — bit for bit — while against
//! `evolve_best_gen0003.bin` it is `1.175e-3`, the whole recorded bias. All five telemetry
//! files pass under the offset-by-one pairing. The rule is what the data says, not an
//! inference from the code alone. A name mismatch prints a HINT and never a failure, and
//! `--tolerancia` exists so a deliberate cross-pairing can still be inspected.
//!
//! # The bias columns are POST-authority
//!
//! `context.rs:539-542` multiplies the organ's output by `HIP_AUTHORITY` (0.5) before the
//! telemetry row is written, so the three recorded columns are the number ADDED to
//! throttle / steer / brake, not `gain · tanh(W·x)`. This tool applies the same factor;
//! dropping it would report a uniform 2× error and look like a broken formula.
//!
//! # What a mismatch can legitimately mean
//!
//! `--hipocampo-apagar` makes the being read `ERASED_READ` (memory zeroed, bias column
//! intact) while the telemetry still records the REAL memory in columns 2..10
//! (`context.rs:2224-2226` reads the slots directly; `hippocampus_bias` reads through
//! `read_gated`). Under that flag a genome with non-zero `W` on the memory columns will
//! disagree here, and the disagreement is the ablation, not a bug.

#[cfg(test)]
mod tests;

use universe::gpu::genome_io::load_genome;
use universe::gpu::hippocampus as hip;

/// Default pass/fail threshold on `max |bias_recomputed − bias_recorded|`.
///
/// Both sides are f32 sums of 9 products through the same function in the same order, so
/// the expected difference is zero and the measured one on all five u0c pairings is
/// exactly 0.0. 1e-5 is slack for a future reader that rounds, not a tolerance this path
/// needs.
const DEFAULT_TOLERANCE: f32 = 1e-5;

/// Column offsets inside one `HIP_OBS_COLS` row. The layout's source of truth is the
/// writer at `src/bin/gpu_evolve/context.rs:2228-2234`; these names are that layout.
const COL_TICK: usize = 0;
const COL_SECTOR: usize = 1;
const COL_MEM: usize = 2;
const COL_BIAS: usize = COL_MEM + 2 * hip::MEM_PER_SECTOR;

/// Largest absolute value in a slice, 0.0 when empty.
///
/// Every statistic in this tool is a peak and not a mean over components: averaging three
/// bias channels of which two are usually near zero hides exactly the one the organ uses.
fn peak(xs: &[f32]) -> f32 {
    xs.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

fn mean(sum: f64, n: usize) -> f64 {
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// One recorded tick of being 0's hippocampus.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HipRow {
    tick: f32,
    sector: usize,
    /// `[mem[s](4), mem[s+1](4)]` — the read WITHOUT the trailing bias column.
    mem: [f32; 2 * hip::MEM_PER_SECTOR],
    /// Post-authority bias, i.e. already × [`hip::HIP_AUTHORITY`].
    bias: [f32; hip::HIP_OUTPUTS],
}

impl HipRow {
    /// The 9-vector `hip::bias` expects: this sector, the next sector, and a constant 1.0.
    fn read(&self) -> [f32; hip::HIP_READ_LEN] {
        let mut x = [0.0f32; hip::HIP_READ_LEN];
        x[..2 * hip::MEM_PER_SECTOR].copy_from_slice(&self.mem);
        x[hip::HIP_READ_LEN - 1] = 1.0;
        x
    }

    /// `mem[s]` — THIS sector's slot, written only when the being LEAVES it, so on a first
    /// pass it is zero by construction. That is the U1 claim, column by column.
    fn here(&self) -> &[f32] {
        &self.mem[..hip::MEM_PER_SECTOR]
    }

    /// `mem[s+1]` — the LOOKAHEAD slot, which CAN be non-zero on a first pass through `s`
    /// and in the u0c data is: the being first reaches sector 23 late in lap 1, when
    /// `mem[0]` has already been written. That is the mechanism working, not a leak.
    fn next(&self) -> &[f32] {
        &self.mem[hip::MEM_PER_SECTOR..]
    }
}

/// Everything that can be wrong with a row, named so the report says WHICH row and WHY.
#[derive(Debug, PartialEq)]
enum RowError {
    /// The file does not hold a whole number of rows — a truncated write, or the wrong
    /// column count (the PRD says 14, the writer writes 13; see `hip::HIP_OBS_COLS`).
    Ragged { floats: usize },
    /// A non-finite value anywhere in the row.
    NotFinite { row: usize, col: usize },
    /// The sector column is not an integer in `0..N_SECTORS`.
    BadSector { row: usize, value: f32 },
}

impl std::fmt::Display for RowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ragged { floats } => write!(
                f,
                "file holds {floats} f32, not a multiple of {} columns \
                 (truncated write, or not a hip_gen<N>.f32)",
                hip::HIP_OBS_COLS
            ),
            Self::NotFinite { row, col } => write!(f, "row {row} column {col} is not finite"),
            Self::BadSector { row, value } => write!(
                f,
                "row {row} sector column is {value}, not an integer in 0..{}",
                hip::N_SECTORS
            ),
        }
    }
}

/// Parses the raw f32 stream of a `hip_gen<N>.f32` into rows, rejecting corruption.
/// Separate from the file read so the tests can feed a synthetic stream.
fn parse_rows(floats: &[f32]) -> Result<Vec<HipRow>, RowError> {
    if floats.len() % hip::HIP_OBS_COLS != 0 {
        return Err(RowError::Ragged {
            floats: floats.len(),
        });
    }
    let mut out = Vec::with_capacity(floats.len() / hip::HIP_OBS_COLS);
    for (r, chunk) in floats.chunks_exact(hip::HIP_OBS_COLS).enumerate() {
        if let Some(c) = chunk.iter().position(|v| !v.is_finite()) {
            return Err(RowError::NotFinite { row: r, col: c });
        }
        let s = chunk[COL_SECTOR];
        if s < 0.0 || s >= hip::N_SECTORS as f32 || s.fract() != 0.0 {
            return Err(RowError::BadSector { row: r, value: s });
        }
        let mut mem = [0.0f32; 2 * hip::MEM_PER_SECTOR];
        mem.copy_from_slice(&chunk[COL_MEM..COL_BIAS]);
        let mut bias = [0.0f32; hip::HIP_OUTPUTS];
        bias.copy_from_slice(&chunk[COL_BIAS..COL_BIAS + hip::HIP_OUTPUTS]);
        out.push(HipRow {
            tick: chunk[COL_TICK],
            sector: s as usize,
            mem,
            bias,
        });
    }
    Ok(out)
}

/// The verdict of `--genoma` + `--hip`.
#[derive(Debug, Default)]
struct CheckReport {
    rows: usize,
    max_diff: f32,
    /// Row index where `max_diff` happened, so a reader can go and look at it.
    worst_row: usize,
    max_abs_bias: f32,
    max_abs_mem: f32,
    sectors_visited: usize,
    /// Rows whose sector differs from the previous row's. Each one is a `cross()`, i.e.
    /// one EMA write into the sector just left.
    crossings: usize,
}

/// Recomputes `HIP_AUTHORITY · bias(genes, x)` for every row and measures the gap.
fn check(genes: &[f32], rows: &[HipRow]) -> CheckReport {
    let mut rep = CheckReport {
        rows: rows.len(),
        ..Default::default()
    };
    let mut seen = [false; hip::N_SECTORS];
    let mut prev_sector: Option<usize> = None;
    for (i, row) in rows.iter().enumerate() {
        let mut want = hip::bias(genes, &row.read());
        for (o, w) in want.iter_mut().enumerate() {
            *w *= hip::HIP_AUTHORITY;
            let d = (*w - row.bias[o]).abs();
            if d > rep.max_diff {
                rep.max_diff = d;
                rep.worst_row = i;
            }
        }
        rep.max_abs_bias = rep.max_abs_bias.max(peak(&row.bias));
        rep.max_abs_mem = rep.max_abs_mem.max(peak(&row.mem));
        seen[row.sector] = true;
        if prev_sector.is_some_and(|p| p != row.sector) {
            rep.crossings += 1;
        }
        prev_sector = Some(row.sector);
    }
    rep.sectors_visited = seen.iter().filter(|&&v| v).count();
    rep
}

/// Per-sector split of the ticks into the FIRST pass through that sector and every later
/// one — the U1 hypothesis, since `mem[s]` is written only when the being leaves `s`.
#[derive(Clone, Copy, Debug, Default)]
struct SectorStat {
    /// Maximal contiguous runs of rows in this sector. A being that wobbles backwards over
    /// a boundary and returns counts two passes, which is what the read actually saw.
    passes: usize,
    ticks: [usize; 2],
    /// `[first pass, later passes]` sums of the per-tick peak, for bias / here / next.
    bias_sum: [f64; 2],
    here_sum: [f64; 2],
    next_sum: [f64; 2],
}

/// Splits every sector's ticks into pass 1 (`[0]`) and passes ≥ 2 (`[1]`).
fn sector_stats(rows: &[HipRow]) -> [SectorStat; hip::N_SECTORS] {
    let mut out = [SectorStat::default(); hip::N_SECTORS];
    let mut prev: Option<usize> = None;
    for row in rows {
        let s = row.sector;
        if prev != Some(s) {
            out[s].passes += 1;
        }
        let k = usize::from(out[s].passes > 1);
        out[s].ticks[k] += 1;
        out[s].bias_sum[k] += peak(&row.bias) as f64;
        out[s].here_sum[k] += peak(row.here()) as f64;
        out[s].next_sum[k] += peak(row.next()) as f64;
        prev = Some(s);
    }
    out
}

/// Generation number in a `hip_gen0003.f32` / `evolve_best_gen0003.bin` name, if any.
/// Used only for the HINT about the elite offset — any other name simply gets no hint.
fn generation_in_name(path: &str) -> Option<u32> {
    let name = std::path::Path::new(path).file_name()?.to_str()?;
    let after = name.rsplit_once("gen")?.1;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The bias a being gets while every memory slot is still zero: `gain · tanh(W[:, 8])`,
/// times the authority. It is NOT zero for an arbitrary genome, which is why both modes
/// print it — "bias on the first pass" is this number, not nothing.
fn resting_bias(genes: &[f32]) -> [f32; hip::HIP_OUTPUTS] {
    let mut b = hip::bias(genes, &hip::ERASED_READ);
    for v in b.iter_mut() {
        *v *= hip::HIP_AUTHORITY;
    }
    b
}

fn usage() -> ! {
    eprintln!(
        "hip_check — verifies the C1 hippocampus formulas against recorded data (no GPU)\n\
         \n\
         ./target/release/hip_check --genoma <evolve_best_genN.bin> --hip <hip_genN.f32> \\\n\
         \x20                         [--tolerancia <f32>]\n\
         ./target/release/hip_check --hip <hip_genN.f32> --setores [--genoma <g.bin>]\n\
         \n\
         NOTE: being 0 of generation N is the ELITE, i.e. the genome of\n\
         evolve_best_gen<N-1>.bin — see this file's header."
    );
    std::process::exit(2);
}

fn next_value(args: &[String], i: usize, flag: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| {
        eprintln!("[ERROR] {flag} requires a value");
        std::process::exit(2);
    })
}

/// Reads a `hip_gen<N>.f32` and parses it, exiting with the right code on failure.
fn load_rows(path: &str) -> Vec<HipRow> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("[ERROR] could not read {path}: {e}");
        std::process::exit(2);
    });
    if bytes.len() % 4 != 0 {
        eprintln!("[FAIL] {path}: {} bytes, not a multiple of 4", bytes.len());
        std::process::exit(1);
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    parse_rows(&floats).unwrap_or_else(|e| {
        eprintln!("[FAIL] {path}: {e}");
        std::process::exit(1);
    })
}

/// Loads the 29 `hip` genes through the canonical parser, exiting 2 if that fails.
fn load_hip_genes(path: &str) -> Vec<f32> {
    let g = load_genome(path).unwrap_or_else(|| {
        eprintln!("[ERROR] could not load genome {path}");
        std::process::exit(2);
    });
    if g.hip.len() != hip::N_HIP_GENES {
        eprintln!(
            "[ERROR] {path}: hip block has {} genes, expected {}",
            g.hip.len(),
            hip::N_HIP_GENES
        );
        std::process::exit(2);
    }
    g.hip
}

/// One line describing the genes, shared by both modes.
fn gene_line(genes: &[f32]) -> String {
    let r = resting_bias(genes);
    format!(
        "alpha = sigmoid(g27) = {:.6}   gain = tanh(g28) = {:.6}   max|W| = {:.6}\n  \
         resting   bias with an EMPTY memory (bias column only): [{:.6}, {:.6}, {:.6}]",
        hip::write_rate(genes),
        hip::read_gain(genes),
        peak(&genes[..hip::GENE_WRITE_RATE]),
        r[0],
        r[1],
        r[2]
    )
}

/// `--genoma` + `--hip`: the recomputation gate.
fn run_check(genome_path: &str, hip_path: &str, tolerance: f32) -> ! {
    let genes = load_hip_genes(genome_path);
    let rows = load_rows(hip_path);
    let rep = check(&genes, &rows);

    println!("hip_check — formula replay, CPU only");
    println!("  genome    {genome_path}");
    println!("  telemetry {hip_path}");
    println!("  genes     {}", gene_line(&genes));
    println!(
        "  rows      {}   ticks {} .. {}",
        rep.rows,
        rows.first().map_or(0.0, |r| r.tick),
        rows.last().map_or(0.0, |r| r.tick)
    );
    println!(
        "  max |bias_recomputed - bias_recorded| = {:.3e}  (worst at row {})",
        rep.max_diff, rep.worst_row
    );
    println!(
        "  max |bias recorded| = {:.6e}   max |mem recorded| = {:.6e}",
        rep.max_abs_bias, rep.max_abs_mem
    );
    let write_fraction = rep.crossings as f32 / rep.rows.max(1) as f32;
    println!(
        "  sectors visited = {}/{}   sector crossings (EMA writes) = {} ({:.3}% of rows)",
        rep.sectors_visited,
        hip::N_SECTORS,
        rep.crossings,
        100.0 * write_fraction
    );

    // The elite offset, printed as a HINT and never as a failure: a deliberate
    // cross-pairing is a legitimate thing to ask this tool for.
    match (
        generation_in_name(genome_path),
        generation_in_name(hip_path),
    ) {
        (Some(_), Some(0)) => println!(
            "  NOTE: generation 0 has no previous elite — being 0 is a random (or \
             --retomar) genome, so no evolve_best_gen*.bin is guaranteed to be it. \
             A PASS here with an all-zero hip block only says both sides are zero."
        ),
        (Some(gg), Some(hg)) if gg + 1 != hg => println!(
            "  HINT: telemetry is generation {hg}, genome is generation {gg}. Being 0 of \
             generation N is the ELITE, carrying the genome of gen N-1 \
             (gpu/evolution.rs:860-873). Try evolve_best_gen{:04}.bin.",
            hg - 1
        ),
        _ => {}
    }

    if rep.max_diff <= tolerance {
        println!("  VERDICT: PASS (tolerance {tolerance:.1e})");
        std::process::exit(0);
    }
    println!(
        "  VERDICT: FAIL (tolerance {tolerance:.1e}). Either this genome is not the being \
         the telemetry recorded (see HINT / the header's elite rule), or the run used \
         --hipocampo-apagar, or the formula changed since the recording."
    );
    std::process::exit(1);
}

/// `--setores`: the per-sector first-pass / later-pass table.
fn run_sectors(hip_path: &str, genome_path: Option<&str>) -> ! {
    let rows = load_rows(hip_path);
    let stats = sector_stats(&rows);

    println!("hip_check --setores — per-sector memory, CPU only");
    println!("  telemetry {hip_path}   rows {}", rows.len());
    if let Some(p) = genome_path {
        println!(
            "  genome    {p}\n  genes     {}",
            gene_line(&load_hip_genes(p))
        );
        println!(
            "  NOTE: the first-pass bias can only be ~0 if the resting bias above is ~0 — \
             the constant 1.0 column of the read is present from tick 0."
        );
    }
    println!(
        "  '1st' = the first contiguous pass through the sector; 'later' = every pass after \
         it. Each figure is the mean over that pass's ticks of the LARGEST of the three \
         bias components (resp. of the four mem[s] values, 'here', and the four mem[s+1] \
         values, 'next'). 'here' is the U1 column: mem[s] is written only when the being \
         LEAVES s, so it must be 0 on a first pass; 'next' may be non-zero there, and that \
         is the lookahead doing its job."
    );
    println!();
    println!(
        "  {:>3} {:>6} {:>7} {:>7} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "sec",
        "passes",
        "t_1st",
        "t_later",
        "|bias|1st",
        "|bias|later",
        "|here|1st",
        "|here|later",
        "|next|1st",
        "|next|later"
    );
    let mut bias_sum = [0.0f64; 2];
    let mut n = [0usize; 2];
    for (s, st) in stats.iter().enumerate() {
        if st.passes == 0 {
            continue;
        }
        println!(
            "  {:>3} {:>6} {:>7} {:>7} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} \
             {:>11.3e} {:>11.3e}",
            s,
            st.passes,
            st.ticks[0],
            st.ticks[1],
            mean(st.bias_sum[0], st.ticks[0]),
            mean(st.bias_sum[1], st.ticks[1]),
            mean(st.here_sum[0], st.ticks[0]),
            mean(st.here_sum[1], st.ticks[1]),
            mean(st.next_sum[0], st.ticks[0]),
            mean(st.next_sum[1], st.ticks[1])
        );
        for k in 0..2 {
            bias_sum[k] += st.bias_sum[k];
            n[k] += st.ticks[k];
        }
    }
    println!(
        "\n  ALL sectors: |bias| first pass {:.6e} over {} ticks · later {:.6e} over {} ticks",
        mean(bias_sum[0], n[0]),
        n[0],
        mean(bias_sum[1], n[1]),
        n[1]
    );
    let visited = stats.iter().filter(|s| s.passes > 0).count();
    let revisited = stats.iter().filter(|s| s.ticks[1] > 0).count();
    println!(
        "  {visited}/{} sectors visited, {revisited} of them entered more than once (0 here \
         means the being never closed a lap, so this file cannot answer the U1 question).",
        hip::N_SECTORS
    );
    std::process::exit(0);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut genome_path: Option<String> = None;
    let mut hip_path: Option<String> = None;
    let mut tolerance = DEFAULT_TOLERANCE;
    let mut sectors_mode = false;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        i += 1;
        match a.as_str() {
            // The flag names match the recording side (`--hipocampo-setor`) and the sibling
            // tool (`sensibilidade --genoma`); renaming them here would split one vocabulary
            // in two. check-language: allow -- CLI flags kept in sync with gpu_evolve
            "--genoma" => {
                genome_path = Some(next_value(&args, i, "--genoma"));
                i += 1;
            }
            "--hip" => {
                hip_path = Some(next_value(&args, i, "--hip"));
                i += 1;
            }
            "--tolerancia" => {
                let v = next_value(&args, i, "--tolerancia");
                i += 1;
                tolerance = v.parse().unwrap_or_else(|_| {
                    eprintln!("[ERROR] --tolerancia expects a number, got '{v}'");
                    std::process::exit(2);
                });
            }
            "--setores" => sectors_mode = true,
            "--help" | "-h" => usage(),
            other => {
                eprintln!("[ERROR] unknown flag '{other}'");
                usage();
            }
        }
    }

    let Some(hip_path) = hip_path else {
        eprintln!("[ERROR] --hip is required");
        usage();
    };
    if sectors_mode {
        run_sectors(&hip_path, genome_path.as_deref());
    }
    let Some(genome_path) = genome_path else {
        eprintln!("[ERROR] --genoma is required unless --setores is given");
        usage();
    };
    run_check(&genome_path, &hip_path, tolerance);
}
