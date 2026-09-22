//! lab-ui PRD S2/I3 -- exports `retina_map.json` (the contract in the I3 task brief):
//! which of the 192 sensory-wave channels reduce retina pixels (and their exact
//! pixel set + op), versus which are body/temporal/whisker/grid-query ("other").
//!
//! Unlike `genome_export`/`track_export`, this map is fully static under the fixed
//! default runtime flags (`perif_real = false`, `BODY_CHANNELS_ON = false`) --
//! see `universe::perception::retina_map`'s module doc -- so it takes no
//! `(run, gen, being, profile)` inputs, just an output path.
//!
//! CPU-only: `universe::perception::retina_map` is pure Rust over `config`/
//! `perception::channels` constants -- no `wgpu` device, no Bevy `App`, no window.
//!
//! Usage:
//!   retina_map_export <out.json>

use std::io::Write as _;

use universe::perception::retina_map::build_retina_map_json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: retina_map_export <out.json>");
        std::process::exit(2);
    }
    let out_path = &args[1];

    let map = build_retina_map_json();
    let text = serde_json::to_string_pretty(&map).expect("serialize retina_map.json");

    let mut f =
        std::fs::File::create(out_path).unwrap_or_else(|e| panic!("cannot create {out_path}: {e}"));
    f.write_all(text.as_bytes())
        .unwrap_or_else(|e| panic!("cannot write {out_path}: {e}"));

    eprintln!("retina_map_export: wrote {out_path} ({} bytes)", text.len());
}
