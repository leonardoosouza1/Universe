// Tier 6.10 — Genetic Engineering Console
// REPL interativo para inspecionar e editar .dna ternários diretamente em disco.
// Zero deps no simulador em runtime — opera exclusivamente sobre arquivos.
//
// Uso: cargo run --release --bin dna_console
// Depois: dna> load CHAMPION.dna  /  info  /  heatmap fast  /  quit
//
// Comandos implementados (versão simplificada):
//   load <path>                        — carrega .dna no slot atual
//   save <path>                        — salva slot atual (escrita atômica)
//   info                               — sumário header + densidade por camada
//   show <layer> <from> <to>           — dump de pesos no intervalo [from, to)
//   set <layer> <idx> <value>          — edita peso único (valor em {-1,0,1})
//   set range <layer> <from> <to> <v>  — edita bulk
//   mutate <rate>                      — flip aleatório de N% pesos (rate 0..1)
//   zero <layer>                       — zera camada inteira
//   scale <layer> <value>              — ajusta weight_scale
//   heatmap <layer>                    — ASCII heatmap estilo dna_view
//   help                               — lista comandos
//   quit | exit                        — sai
//
// <layer> ∈ {fast, slow}

use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use universe::evolution::dna::{
    load_champion_dna, save_champion_dna, DnaPayload, DNA_FAST_SIZE, DNA_SLOW_SIZE,
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
    println!("Digite `help` para listar comandos, `quit` pra sair.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut slot: Option<Slot> = None;

    // Loop REPL linha-a-linha.
    let mut handle = stdin.lock();
    loop {
        print!("dna> ");
        let _ = stdout.flush();
        let mut line = String::new();
        match handle.read_line(&mut line) {
            Ok(0) => { println!(); break; }   // EOF
            Ok(_) => {}
            Err(e) => { eprintln!("erro lendo stdin: {}", e); break; }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        let cmd = tokens[0].to_lowercase();
        let args = &tokens[1..];

        match cmd.as_str() {
            "quit" | "exit" | "q" => {
                if let Some(ref s) = slot {
                    if s.dirty {
                        println!("⚠  slot com edições não salvas (last-loaded: {}). Saindo mesmo assim.", s.source_path);
                    }
                }
                break;
            }
            "help" | "?" => print_help(),
            "load" => {
                if args.len() != 1 { println!("uso: load <path>"); continue; }
                match load_champion_dna(args[0]) {
                    Some(dna) => {
                        println!("✔ carregado {} (v{}, gen {}, {} pesos fast / {} slow)",
                            args[0], dna.version, dna.generation,
                            dna.fast_weights.len(),
                            dna.slow_weights.as_ref().map(|s| s.len()).unwrap_or(0));
                        slot = Some(Slot { dna, source_path: args[0].to_string(), dirty: false });
                    }
                    None => println!("❌ falha ao carregar {} (inexistente ou corrompido)", args[0]),
                }
            }
            "save" => {
                if args.len() != 1 { println!("uso: save <path>"); continue; }
                let Some(s) = slot.as_mut() else { println!("nenhum DNA no slot; use `load` primeiro"); continue; };
                match atomic_save(args[0], &s.dna) {
                    Ok(()) => { println!("✔ salvo em {}", args[0]); s.dirty = false; }
                    Err(e) => println!("❌ erro salvando: {}", e),
                }
            }
            "info" => {
                let Some(s) = slot.as_ref() else { println!("nenhum DNA no slot"); continue; };
                print_info(&s.dna, &s.source_path, s.dirty);
            }
            "show" => {
                if args.len() != 3 { println!("uso: show <fast|slow> <from> <to>"); continue; }
                let Some(s) = slot.as_ref() else { println!("nenhum DNA no slot"); continue; };
                let Some(layer) = pick_layer(&s.dna, args[0]) else { println!("layer inválida (use fast|slow)"); continue; };
                let (from, to) = match (args[1].parse::<usize>(), args[2].parse::<usize>()) {
                    (Ok(a), Ok(b)) => (a, b),
                    _ => { println!("from/to devem ser usize"); continue; }
                };
                if from >= layer.len() || to > layer.len() || from >= to {
                    println!("intervalo inválido (layer tem {} pesos)", layer.len()); continue;
                }
                print!("[{}..{})", from, to);
                for (i, &w) in layer[from..to].iter().enumerate() {
                    if i % 32 == 0 { println!(); print!("  {:>5}: ", from + i); }
                    print!("{:>2} ", w);
                }
                println!();
            }
            "set" => {
                let Some(s) = slot.as_mut() else { println!("nenhum DNA no slot"); continue; };
                // set range <layer> <from> <to> <value>
                if args.first().map(|a| a.eq_ignore_ascii_case("range")).unwrap_or(false) {
                    if args.len() != 5 { println!("uso: set range <fast|slow> <from> <to> <value>"); continue; }
                    let Some(value) = parse_ternary(args[4]) else { println!("value deve ser -1, 0 ou 1"); continue; };
                    let (from, to) = match (args[2].parse::<usize>(), args[3].parse::<usize>()) {
                        (Ok(a), Ok(b)) => (a, b),
                        _ => { println!("from/to devem ser usize"); continue; }
                    };
                    let layer = match pick_layer_mut(&mut s.dna, args[1]) {
                        Some(l) => l, None => { println!("layer inválida"); continue; }
                    };
                    if from >= layer.len() || to > layer.len() || from >= to {
                        println!("intervalo inválido (layer tem {} pesos)", layer.len()); continue;
                    }
                    for w in &mut layer[from..to] { *w = value; }
                    s.dirty = true;
                    println!("✔ set range {}[{}..{}) = {}", args[1], from, to, value);
                    continue;
                }
                // set <layer> <idx> <value>
                if args.len() != 3 { println!("uso: set <fast|slow> <idx> <value>  OU  set range <layer> <from> <to> <value>"); continue; }
                let Some(value) = parse_ternary(args[2]) else { println!("value deve ser -1, 0 ou 1"); continue; };
                let idx: usize = match args[1].parse() { Ok(i) => i, Err(_) => { println!("idx inválido"); continue; } };
                let layer = match pick_layer_mut(&mut s.dna, args[0]) {
                    Some(l) => l, None => { println!("layer inválida"); continue; }
                };
                if idx >= layer.len() { println!("idx fora do range (layer tem {} pesos)", layer.len()); continue; }
                let old = layer[idx];
                layer[idx] = value;
                s.dirty = true;
                println!("✔ {}[{}] : {} → {}", args[0], idx, old, value);
            }
            "mutate" => {
                if args.len() != 1 { println!("uso: mutate <rate 0..1>"); continue; }
                let Some(s) = slot.as_mut() else { println!("nenhum DNA no slot"); continue; };
                let rate: f32 = match args[0].parse() { Ok(r) => r, Err(_) => { println!("rate inválido"); continue; } };
                if !(0.0..=1.0).contains(&rate) { println!("rate deve estar em [0,1]"); continue; }
                let fast_flips = random_flip(&mut s.dna.fast_weights, rate, 0xC0FFEE);
                let slow_flips = if let Some(ref mut sw) = s.dna.slow_weights {
                    random_flip(sw, rate, 0xBADCAB1E)
                } else { 0 };
                s.dirty = true;
                println!("✔ mutate rate={:.4} — flips fast={} slow={}", rate, fast_flips, slow_flips);
            }
            "zero" => {
                if args.len() != 1 { println!("uso: zero <fast|slow>"); continue; }
                let Some(s) = slot.as_mut() else { println!("nenhum DNA no slot"); continue; };
                let layer = match pick_layer_mut(&mut s.dna, args[0]) {
                    Some(l) => l, None => { println!("layer inválida"); continue; }
                };
                let n = layer.len();
                for w in layer { *w = 0; }
                s.dirty = true;
                println!("✔ zerou {} pesos em {}", n, args[0]);
            }
            "scale" => {
                if args.len() != 2 { println!("uso: scale <fast|slow> <value>"); continue; }
                let Some(s) = slot.as_mut() else { println!("nenhum DNA no slot"); continue; };
                let value: f32 = match args[1].parse() { Ok(v) => v, Err(_) => { println!("value inválido"); continue; } };
                match args[0].to_lowercase().as_str() {
                    "fast" => { let old = s.dna.fast_scale; s.dna.fast_scale = value; println!("✔ fast_scale: {:.6} → {:.6}", old, value); }
                    "slow" => { let old = s.dna.slow_scale; s.dna.slow_scale = value; println!("✔ slow_scale: {:.6} → {:.6}", old, value); }
                    _ => { println!("layer inválida"); continue; }
                }
                s.dirty = true;
            }
            "heatmap" => {
                if args.len() != 1 { println!("uso: heatmap <fast|slow>"); continue; }
                let Some(s) = slot.as_ref() else { println!("nenhum DNA no slot"); continue; };
                match args[0].to_lowercase().as_str() {
                    "fast" => heatmap("INSTINCT FAST (128 inputs × 64 neurons)", &s.dna.fast_weights, 128, 64, DNA_FAST_SIZE),
                    "slow" => match s.dna.slow_weights.as_ref() {
                        Some(sw) => heatmap("SLOW BRAIN (32 inputs × 8 neurons)", sw, 32, 8, DNA_SLOW_SIZE),
                        None => println!("⚠ slow_weights ausente neste DNA"),
                    },
                    _ => println!("layer inválida"),
                }
            }
            other => println!("comando desconhecido: `{}` (tente `help`)", other),
        }
    }

    println!("bye.");
}

fn print_help() {
    println!("comandos:");
    println!("  load <path>                        carrega .dna no slot");
    println!("  save <path>                        salva slot (atomic)");
    println!("  info                               sumário do slot atual");
    println!("  show <fast|slow> <from> <to>       dump de pesos");
    println!("  set <fast|slow> <idx> <-1|0|1>     edita um peso");
    println!("  set range <layer> <from> <to> <v>  edita bulk");
    println!("  mutate <rate>                      flip aleatório (0..1)");
    println!("  zero <fast|slow>                   zera camada");
    println!("  scale <fast|slow> <value>          ajusta weight_scale");
    println!("  heatmap <fast|slow>                ASCII heatmap");
    println!("  quit | exit                        sai");
}

fn print_info(d: &DnaPayload, src: &str, dirty: bool) {
    println!("─────────────────────────────────────────────────────────────────");
    println!("📋 slot <- {} {}", src, if dirty { "(DIRTY — não salvo)" } else { "" });
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
        println!("  slow          : ausente (legacy v1)");
    }
}

fn print_density(label: &str, weights: &[i8]) {
    if weights.is_empty() { println!("  {:<14}: vazio", label); return; }
    let (mut n, mut z, mut p) = (0i64, 0i64, 0i64);
    for &w in weights { if w < 0 { n += 1 } else if w == 0 { z += 1 } else { p += 1 } }
    let t = weights.len() as f32;
    println!("  {:<14}: {} pesos | -1:{:5} ({:.1}%)  0:{:5} ({:.1}%)  +1:{:5} ({:.1}%)  densidade {:.1}%",
        label, weights.len(),
        n, n as f32 * 100.0 / t,
        z, z as f32 * 100.0 / t,
        p, p as f32 * 100.0 / t,
        (n + p) as f32 * 100.0 / t);
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
        -1 => Some(-1), 0 => Some(0), 1 => Some(1), _ => None,
    }
}

// LCG determinístico para flips — evita depender de rand em um binário-utilitário.
fn random_flip(weights: &mut [i8], rate: f32, seed_bias: u64) -> usize {
    let mut state: u64 = seed_bias.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(weights.len() as u64);
    let mut flips = 0usize;
    for w in weights.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        if r < rate {
            // Escolhe novo valor ternário ≠ atual.
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = ((state >> 40) as u32) % 3;
            let new_v: i8 = match pick { 0 => -1, 1 => 0, _ => 1 };
            if new_v != *w { *w = new_v; flips += 1; }
        }
    }
    flips
}

// Escrita atômica: escreve em <path>.tmp e depois rename. Evita corrupção se o
// processo morrer no meio da serialização. Reutiliza save_champion_dna.
fn atomic_save(path: &str, d: &DnaPayload) -> io::Result<()> {
    let tmp = format!("{}.tmp", path);
    // slow_weights pode estar ausente em DNAs legacy v1; materializa zeros.
    let slow_ref: Vec<i8> = match d.slow_weights.as_ref() {
        Some(s) => s.clone(),
        None => vec![0i8; DNA_SLOW_SIZE],
    };
    save_champion_dna(
        &tmp,
        &d.fast_weights, d.fast_scale,
        &slow_ref, d.slow_scale,
        d.generation, d.seed, d.track_width, d.fitness,
    )?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn heatmap(title: &str, weights: &[i8], inputs: usize, neurons: usize, expected: usize) {
    println!("─────────────────────────────────────────────────────────────────");
    println!("🔹 {}", title);
    println!("  esperado: {}  real: {}", expected, weights.len());
    if weights.is_empty() { println!("  ⚠ vazio"); return; }
    println!("  legenda: '-'=inibe  '.'=mudo  '+'=excita");
    let samples = neurons.min(16);
    for n in 0..samples {
        print!("  N{:02} │", n);
        for i in 0..inputs.min(64) {
            let idx = n * inputs + i;
            if idx >= weights.len() { break; }
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
        println!("  … ({} neurônios adicionais omitidos)", neurons - samples);
    }
}

// Silencia unused import (File/Path só aparecem via save_champion_dna, mas o trait
// Write é usado diretamente no flush do prompt).
#[allow(dead_code)] fn _keep_imports(_f: File, _p: &Path) {}
