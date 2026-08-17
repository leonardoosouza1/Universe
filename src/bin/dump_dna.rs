//! `dump_dna` — what is inside a `.dna`, before trusting any read of it.
//!
//! Exists because a v8 `.dna` stores the layer-1 weights in TWO places with different
//! scales, and reading the wrong one raises no error — it yields a silently false report:
//!
//!   `fast_weights`  TERNARY {−1,0,+1} version from the compressor. Divided by 63.5 it becomes ±0.016
//!                   and makes the whole brain look empty.
//!   chunk `CRAW`    the real i8 weights, ±127 → ±2.0. It is what `spawn.rs:142` prefers.
//!
//! That mistake is what made `sinapses` report "0 synapses" on a champion that had 1637.
//! Running this first costs one second and shows which source the file really has.
//!
//! ```text
//! cargo run --release --bin dump_dna -- CHAMPION.dna
//! ```

use universe::evolution::dna_genesis::load_champion_dna;

fn perfil(rotulo: &str, w: &[i8]) {
    if w.is_empty() {
        println!("  {rotulo:<14} empty");
        return;
    }
    let nz = w.iter().filter(|&&x| x != 0).count();
    let (mn, mx) = w.iter().fold((0i8, 0i8), |(a, b), &x| (a.min(x), b.max(x)));
    // 8 bins of 16 over the absolute value: reveals at once whether the content is ternary
    // (everything in the first bin) or full i8 (with a tail up to 127)
    let mut h = [0usize; 8];
    for &x in w {
        h[(((x as i32).abs() / 16) as usize).min(7)] += 1;
    }
    let ternario = mx <= 1 && mn >= -1;
    println!(
        "  {rotulo:<14} n={:<7} non-zero={:<6} ({:>5.1}%)  range=[{:>4},{:>4}]  {}",
        w.len(),
        nz,
        nz as f32 / w.len() as f32 * 100.0,
        mn,
        mx,
        if ternario {
            "TERNARY ⚠️  /63.5 here yields ±0.016"
        } else {
            "full i8"
        }
    );
    println!("  {:<14} |i8| histogram in bins of 16: {:?}", "", h);
}

fn main() {
    let arquivo = match std::env::args().nth(1) {
        Some(a) => a,
        None => {
            eprintln!("usage: dump_dna <file.dna>");
            std::process::exit(2);
        }
    };
    let d = match load_champion_dna(&arquivo) {
        Some(d) => d,
        None => {
            eprintln!("could not read {arquivo} (unrecognized format?)");
            std::process::exit(1);
        }
    };

    println!("=== {arquivo} ===");
    println!(
        "version={}  generation={}  fitness={:.0}  seed={}  track width={:.1}m",
        d.version, d.generation, d.fitness, d.seed, d.track_width
    );
    println!("fast_scale={}  slow_scale={}", d.fast_scale, d.slow_scale);

    println!("\nLAYER 1 WEIGHTS — the two possible sources:");
    perfil("fast_weights", &d.fast_weights);
    match d.cortex_raw.as_ref() {
        Some(c) => perfil("CRAW", c),
        None => println!("  {:<14} absent", "CRAW"),
    }

    println!("\nOTHER BLOCKS:");
    let opt =
        |v: Option<&Vec<i8>>| v.map_or("absent".to_string(), |x| format!("{} weights", x.len()));
    println!("  cortex_l2_raw  {}", opt(d.cortex_l2_raw.as_ref()));
    println!("  slow_weights   {}", opt(d.slow_weights.as_ref()));
    println!(
        "  trinity        {}",
        if d.trinity.is_some() {
            "present (gate + reflex)"
        } else {
            "absent"
        }
    );

    let fonte = if d.cortex_raw.is_some() {
        "CRAW"
    } else {
        "fast_weights"
    };
    println!("\n-> `sinapses` will read: {fonte}");
    if d.cortex_raw.is_none() {
        println!("   ⚠️  without CRAW the weights come out ~64× smaller than they are.");
    }
}
