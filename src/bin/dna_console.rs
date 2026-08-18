// Tier 6.10 — Genetic Engineering Console
// Interactive REPL to inspect and edit ternary .dna files directly on disk.
// Zero deps on the simulator at runtime — operates exclusively on files.
//
// Usage: cargo run --release --bin dna_console
// Then: dna> load CHAMPION.dna  /  info  /  heatmap fast  /  quit
//
// Implemented commands (simplified version):
//   load <path>                        — loads .dna into the current slot
//   save <path>                        — saves current slot (atomic write)
//   info                               — header summary + density per layer
//   show <layer> <from> <to>           — weight dump in the range [from, to)
//   set <layer> <idx> <value>          — edits a single weight (value in {-1,0,1})
//   set range <layer> <from> <to> <v>  — bulk edit
//   mutate <rate>                      — random flip of N% weights (rate 0..1)
//   zero <layer>                       — zeroes the whole layer
//   scale <layer> <value>              — adjusts weight_scale
//   heatmap <layer>                    — ASCII heatmap in dna_view style
//   help                               — lists commands
//   quit | exit                        — exits
//
// <layer> ∈ {fast, slow}

use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use universe::evolution::dna_genesis::{
    DNA_FAST_SIZE, DNA_SLOW_SIZE, DnaPayload, load_champion_dna, save_champion_dna,
};

struct Slot {
    dna: DnaPayload,
    source_path: String,
    dirty: bool,
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("🧬 DNA CONSOLE — Tier 6.10 Genetic Engineering REPL");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Type `help` to list commands, `quit` to exit.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut slot: Option<Slot> = None;

    // Line-by-line REPL loop.
    let mut handle = stdin.lock();
    loop {
        print!("dna> ");
        let _ = stdout.flush();
        let mut line = String::new();
        match handle.read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            } // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("error reading stdin: {}", e);
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        let cmd = tokens[0].to_lowercase();
        let args = &tokens[1..];

        match cmd.as_str() {
            "quit" | "exit" | "q" => {
                if let Some(ref s) = slot {
                    if s.dirty {
                        println!(
                            "⚠  slot has unsaved edits (last-loaded: {}). Exiting anyway.",
                            s.source_path
                        );
                    }
                }
                break;
            }
            "help" | "?" => print_help(),
            "load" => {
                if args.len() != 1 {
                    println!("usage: load <path>");
                    continue;
                }
                match load_champion_dna(args[0]) {
                    Some(dna) => {
                        println!(
                            "✔ loaded {} (v{}, gen {}, {} fast weights / {} slow)",
                            args[0],
                            dna.version,
                            dna.generation,
                            dna.fast_weights.len(),
                            dna.slow_weights.as_ref().map(|s| s.len()).unwrap_or(0)
                        );
                        slot = Some(Slot {
                            dna,
                            source_path: args[0].to_string(),
                            dirty: false,
                        });
                    }
                    None => println!("❌ failed to load {} (missing or corrupted)", args[0]),
                }
            }
            "save" => {
                if args.len() != 1 {
                    println!("usage: save <path>");
                    continue;
                }
                let Some(s) = slot.as_mut() else {
                    println!("no DNA in the slot; use `load` first");
                    continue;
                };
                match atomic_save(args[0], &s.dna) {
                    Ok(()) => {
                        println!("✔ saved to {}", args[0]);
                        s.dirty = false;
                    }
                    Err(e) => println!("❌ error saving: {}", e),
                }
            }
            "info" => {
                let Some(s) = slot.as_ref() else {
                    println!("no DNA in the slot");
                    continue;
                };
                print_info(&s.dna, &s.source_path, s.dirty);
            }
            "show" => {
                if args.len() != 3 {
                    println!("usage: show <fast|slow> <from> <to>");
                    continue;
                }
                let Some(s) = slot.as_ref() else {
                    println!("no DNA in the slot");
                    continue;
                };
                let Some(layer) = pick_layer(&s.dna, args[0]) else {
                    println!("invalid layer (use fast|slow)");
                    continue;
                };
                let (from, to) = match (args[1].parse::<usize>(), args[2].parse::<usize>()) {
                    (Ok(a), Ok(b)) => (a, b),
                    _ => {
                        println!("from/to must be usize");
                        continue;
                    }
                };
                if from >= layer.len() || to > layer.len() || from >= to {
                    println!("invalid range (layer has {} weights)", layer.len());
                    continue;
                }
                print!("[{}..{})", from, to);
                for (i, &w) in layer[from..to].iter().enumerate() {
                    if i % 32 == 0 {
                        println!();
                        print!("  {:>5}: ", from + i);
                    }
                    print!("{:>2} ", w);
                }
                println!();
            }
            "set" => {
                let Some(s) = slot.as_mut() else {
                    println!("no DNA in the slot");
                    continue;
                };
                // set range <layer> <from> <to> <value>
                if args
                    .first()
                    .map(|a| a.eq_ignore_ascii_case("range"))
                    .unwrap_or(false)
                {
                    if args.len() != 5 {
                        println!("usage: set range <fast|slow> <from> <to> <value>");
                        continue;
                    }
                    let Some(value) = parse_ternary(args[4]) else {
                        println!("value must be -1, 0 or 1");
                        continue;
                    };
                    let (from, to) = match (args[2].parse::<usize>(), args[3].parse::<usize>()) {
                        (Ok(a), Ok(b)) => (a, b),
                        _ => {
                            println!("from/to must be usize");
                            continue;
                        }
                    };
                    let layer = match pick_layer_mut(&mut s.dna, args[1]) {
                        Some(l) => l,
                        None => {
                            println!("invalid layer");
                            continue;
                        }
                    };
                    if from >= layer.len() || to > layer.len() || from >= to {
                        println!("invalid range (layer has {} weights)", layer.len());
                        continue;
                    }
                    for w in &mut layer[from..to] {
                        *w = value;
                    }
                    s.dirty = true;
                    println!("✔ set range {}[{}..{}) = {}", args[1], from, to, value);
                    continue;
                }
                // set <layer> <idx> <value>
                if args.len() != 3 {
                    println!(
                        "usage: set <fast|slow> <idx> <value>  OR  set range <layer> <from> <to> <value>"
                    );
                    continue;
                }
                let Some(value) = parse_ternary(args[2]) else {
                    println!("value must be -1, 0 or 1");
                    continue;
                };
                let idx: usize = match args[1].parse() {
                    Ok(i) => i,
                    Err(_) => {
                        println!("invalid idx");
                        continue;
                    }
                };
                let layer = match pick_layer_mut(&mut s.dna, args[0]) {
                    Some(l) => l,
                    None => {
                        println!("invalid layer");
                        continue;
                    }
                };
                if idx >= layer.len() {
                    println!("idx out of range (layer has {} weights)", layer.len());
                    continue;
                }
                let old = layer[idx];
                layer[idx] = value;
                s.dirty = true;
                println!("✔ {}[{}] : {} → {}", args[0], idx, old, value);
            }
            "mutate" => {
                if args.len() != 1 {
                    println!("usage: mutate <rate 0..1>");
                    continue;
                }
                let Some(s) = slot.as_mut() else {
                    println!("no DNA in the slot");
                    continue;
                };
                let rate: f32 = match args[0].parse() {
                    Ok(r) => r,
                    Err(_) => {
                        println!("invalid rate");
                        continue;
                    }
                };
                if !(0.0..=1.0).contains(&rate) {
                    println!("rate must be in [0,1]");
                    continue;
                }
                let fast_flips = random_flip(&mut s.dna.fast_weights, rate, 0xC0FFEE);
                let slow_flips = if let Some(ref mut sw) = s.dna.slow_weights {
                    random_flip(sw, rate, 0xBADCAB1E)
                } else {
                    0
                };
                s.dirty = true;
                println!(
                    "✔ mutate rate={:.4} — flips fast={} slow={}",
                    rate, fast_flips, slow_flips
                );
            }
            "zero" => {
                if args.len() != 1 {
                    println!("usage: zero <fast|slow>");
                    continue;
                }
                let Some(s) = slot.as_mut() else {
                    println!("no DNA in the slot");
                    continue;
                };
                let layer = match pick_layer_mut(&mut s.dna, args[0]) {
                    Some(l) => l,
                    None => {
                        println!("invalid layer");
                        continue;
                    }
                };
                let n = layer.len();
                for w in layer {
                    *w = 0;
                }
                s.dirty = true;
                println!("✔ zeroed {} weights in {}", n, args[0]);
            }
            "scale" => {
                if args.len() != 2 {
                    println!("usage: scale <fast|slow> <value>");
                    continue;
                }
                let Some(s) = slot.as_mut() else {
                    println!("no DNA in the slot");
                    continue;
                };
                let value: f32 = match args[1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        println!("invalid value");
                        continue;
                    }
                };
                match args[0].to_lowercase().as_str() {
                    "fast" => {
                        let old = s.dna.fast_scale;
                        s.dna.fast_scale = value;
                        println!("✔ fast_scale: {:.6} → {:.6}", old, value);
                    }
                    "slow" => {
                        let old = s.dna.slow_scale;
                        s.dna.slow_scale = value;
                        println!("✔ slow_scale: {:.6} → {:.6}", old, value);
                    }
                    _ => {
                        println!("invalid layer");
                        continue;
                    }
                }
                s.dirty = true;
            }
            "heatmap" => {
                if args.len() != 1 {
                    println!("usage: heatmap <fast|slow>");
                    continue;
                }
                let Some(s) = slot.as_ref() else {
                    println!("no DNA in the slot");
                    continue;
                };
                match args[0].to_lowercase().as_str() {
                    "fast" => heatmap(
                        "INSTINCT FAST (128 inputs × 64 neurons)",
                        &s.dna.fast_weights,
                        128,
                        64,
                        DNA_FAST_SIZE,
                    ),
                    "slow" => match s.dna.slow_weights.as_ref() {
                        Some(sw) => heatmap(
                            "SLOW BRAIN (32 inputs × 8 neurons)",
                            sw,
                            32,
                            8,
                            DNA_SLOW_SIZE,
                        ),
                        None => println!("⚠ slow_weights missing in this DNA"),
                    },
                    _ => println!("invalid layer"),
                }
            }
            other => println!("unknown command: `{}` (try `help`)", other),
        }
    }

    println!("bye.");
}

fn print_help() {
    println!("commands:");
    println!("  load <path>                        loads .dna into the slot");
    println!("  save <path>                        saves the slot (atomic)");
    println!("  info                               summary of the current slot");
    println!("  show <fast|slow> <from> <to>       weight dump");
    println!("  set <fast|slow> <idx> <-1|0|1>     edits one weight");
    println!("  set range <layer> <from> <to> <v>  bulk edit");
    println!("  mutate <rate>                      random flip (0..1)");
    println!("  zero <fast|slow>                   zeroes the layer");
    println!("  scale <fast|slow> <value>          adjusts weight_scale");
    println!("  heatmap <fast|slow>                ASCII heatmap");
    println!("  quit | exit                        exits");
}

fn print_info(d: &DnaPayload, src: &str, dirty: bool) {
    println!("─────────────────────────────────────────────────────────────────");
    println!(
        "📋 slot <- {} {}",
        src,
        if dirty { "(DIRTY — not saved)" } else { "" }
    );
    println!("  version       : v{}", d.version);
    println!("  generation    : {}", d.generation);
    println!("  seed          : {}", d.seed);
    println!("  track_width   : {:.2} m", d.track_width);
    println!("  fitness       : {:.2}", d.fitness);
    println!("  fast_scale    : {:.6}", d.fast_scale);
    println!("  slow_scale    : {:.6}", d.slow_scale);
    print_density("fast", &d.fast_weights);
    if let Some(ref sw) = d.slow_weights {
        print_density("slow", sw);
    } else {
        println!("  slow          : missing (legacy v1)");
    }
}

fn print_density(label: &str, weights: &[i8]) {
    if weights.is_empty() {
        println!("  {:<14}: empty", label);
        return;
    }
    let (mut n, mut z, mut p) = (0i64, 0i64, 0i64);
    for &w in weights {
        if w < 0 {
            n += 1
        } else if w == 0 {
            z += 1
        } else {
            p += 1
        }
    }
    let t = weights.len() as f32;
    println!(
        "  {:<14}: {} weights | -1:{:5} ({:.1}%)  0:{:5} ({:.1}%)  +1:{:5} ({:.1}%)  density {:.1}%",
        label,
        weights.len(),
        n,
        n as f32 * 100.0 / t,
        z,
        z as f32 * 100.0 / t,
        p,
        p as f32 * 100.0 / t,
        (n + p) as f32 * 100.0 / t
    );
}

fn pick_layer<'a>(d: &'a DnaPayload, name: &str) -> Option<&'a [i8]> {
    match name.to_lowercase().as_str() {
        "fast" => Some(&d.fast_weights),
        "slow" => d.slow_weights.as_deref(),
        _ => None,
    }
}

fn pick_layer_mut<'a>(d: &'a mut DnaPayload, name: &str) -> Option<&'a mut [i8]> {
    match name.to_lowercase().as_str() {
        "fast" => Some(&mut d.fast_weights),
        "slow" => d.slow_weights.as_deref_mut(),
        _ => None,
    }
}

fn parse_ternary(s: &str) -> Option<i8> {
    match s.parse::<i32>().ok()? {
        -1 => Some(-1),
        0 => Some(0),
        1 => Some(1),
        _ => None,
    }
}

// Deterministic LCG for flips — avoids depending on rand in a utility binary.
fn random_flip(weights: &mut [i8], rate: f32, seed_bias: u64) -> usize {
    let mut state: u64 = seed_bias
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(weights.len() as u64);
    let mut flips = 0usize;
    for w in weights.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        if r < rate {
            // Picks a new ternary value ≠ the current one.
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let pick = ((state >> 40) as u32) % 3;
            let new_v: i8 = match pick {
                0 => -1,
                1 => 0,
                _ => 1,
            };
            if new_v != *w {
                *w = new_v;
                flips += 1;
            }
        }
    }
    flips
}

// Atomic write: writes to <path>.tmp and then renames. Avoids corruption if the
// process dies mid-serialization. Reuses save_champion_dna.
fn atomic_save(path: &str, d: &DnaPayload) -> io::Result<()> {
    let tmp = format!("{}.tmp", path);
    // slow_weights may be missing in legacy v1 DNAs; materializes zeros.
    let slow_ref: Vec<i8> = match d.slow_weights.as_ref() {
        Some(s) => s.clone(),
        None => vec![0i8; DNA_SLOW_SIZE],
    };
    save_champion_dna(
        &tmp,
        &d.fast_weights,
        d.fast_scale,
        &slow_ref,
        d.slow_scale,
        d.generation,
        d.seed,
        d.track_width,
        d.fitness,
    )?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn heatmap(title: &str, weights: &[i8], inputs: usize, neurons: usize, expected: usize) {
    println!("─────────────────────────────────────────────────────────────────");
    println!("🔹 {}", title);
    println!("  expected: {}  actual: {}", expected, weights.len());
    if weights.is_empty() {
        println!("  ⚠ empty");
        return;
    }
    println!("  legend: '-'=inhibits  '.'=mute  '+'=excites");
    let samples = neurons.min(16);
    for n in 0..samples {
        print!("  N{:02} │", n);
        for i in 0..inputs.min(64) {
            let idx = n * inputs + i;
            if idx >= weights.len() {
                break;
            }
            let c = match weights[idx] {
                w if w < 0 => '-',
                0 => '.',
                _ => '+',
            };
            print!("{}", c);
        }
        println!();
    }
    if samples < neurons {
        println!("  … ({} additional neurons omitted)", neurons - samples);
    }
}

// Silences unused import (File/Path only appear via save_champion_dna, but the Write
// trait is used directly in the prompt flush).
#[allow(dead_code)]
fn _keep_imports(_f: File, _p: &Path) {}
