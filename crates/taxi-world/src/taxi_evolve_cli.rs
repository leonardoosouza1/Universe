//! `taxi_evolve` (T7): the CLI logic that wires the lab's own scenario/champion
//! JSON files to this crate's `evolution::evolve` + `judge::judge`, over
//! `sim_cpu::SimCpu`, and writes the result in the SAME schema
//! `scripts/analise_campo.py`/`src/retrato_bateria.py` (the lab, Python) already
//! read — so a Rust run drops into the same directory as a Python `e43_campo.py`
//! run with no changes on that side.
//!
//! Lives in the LIBRARY, not `src/bin/taxi_evolve.rs`, on purpose: a Rust
//! integration test (`tests/*.rs`) can only link a package's LIBRARY target,
//! never another binary target — so `run` is `pub` here and `src/bin/
//! taxi_evolve.rs` is a thin wrapper calling straight into it, which is what
//! lets `tests/evolve_smoke.rs` drive the full CLI "in code, not
//! a subprocess" (this task's own acceptance-test wording).
//!
//! ```text
//! taxi_evolve --cenario <cenario.json> --mapa <traco_ref.json | dir/> \
//!             --campeao <e43_*.json> [--saida <path>] [--tag <str>] \
//!             [--seeds 0 1 2] [--P N] [--G N] [--R N] [--sigma-mut F] [--paciencia N]
//! ```
//!
//! - `--cenario`: a lab scenario (`neuromod-lab/cenarios/*.json`, e.g.
//!   `mundoA_seres.json`) — `protocolo` (P/G/paciencia/R/seeds/sigma_mut/campeao/
//!   crn) and `mundo`/`corpo`/`passageiros` (mapped onto `Config` where this
//!   crate has an equivalent field — see `build_config`). Only the `linear` arm
//!   makes sense in this v0 (no non-Linear brain exists in this crate); any
//!   other name in `cenario.bracos` is skipped with a warning, never silently.
//! - `--mapa`: a single trace JSON (v0's only map source, SPEC §2.2 — this crate
//!   never GENERATES a map) or a directory of them (`NC > 1`, SPEC §10's `mapas`
//!   axis). **The judge (SPEC §9) is supposed to run on maps DIFFERENT from
//!   training** (`gerar_mapa(1000+s)` vs `gerar_mapa(0..3)`); this CLI has no
//!   separate `--mapa-teste` flag (not in this task's own flag list) and reuses
//!   `--mapa` for both — a declared v0 simplification, not a SPEC-faithful
//!   `medidas` call. See this crate's final report.
//! - `--campeao`: an `e43_*.json` output (Python OR a prior `taxi_evolve` run)
//!   supplying `bracos.linear.W[seed]` — 40 floats in `nomes_f` order — used as
//!   `GaParams::seed_from` (`--campeao` seeds the population, it does not just
//!   get carried through unevolved). A seed the file does not have falls back to
//!   another seed's champion, cyclically (mirrors `e43_campo.py`'s own
//!   `alinha_semente` fallback for a champion file with fewer seeds than
//!   requested).
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::evolution::{Champion, ChampionMode, GaParams, evolve};
use crate::judge::{Measures, judge, measures_to_json};
use crate::sim_cpu::SimCpu;
use crate::trace::Trace;
use crate::world::World;
use crate::{Config, Genome};

/// Runs the whole CLI given its argv (without `argv[0]`) and returns the path
/// it wrote on success. Never calls `std::process::exit` — that is the
/// binary's (`src/bin/taxi_evolve.rs`) job, so a test can inspect the `Err`
/// instead of the process dying.
pub fn run(args: &[String]) -> Result<PathBuf, String> {
    let raw = RawArgs::parse(args)?;

    let cenario_path = raw.required_path("cenario")?;
    let mapa_path = raw.required_path("mapa")?;
    let campeao_path = raw.required_path("campeao")?;
    let tag = raw.one("tag").unwrap_or("").to_string();
    let saida_path = raw.one("saida").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(format!(
            "/media/llm-data/universe-runs/neuromod-lab-taxi/e82/e43{tag}.json"
        ))
    });

    let cenario_text = fs::read_to_string(&cenario_path)
        .map_err(|e| format!("could not read --cenario {}: {e}", cenario_path.display()))?;
    let cenario: Value = serde_json::from_str(&cenario_text).map_err(|e| {
        format!(
            "could not parse --cenario JSON {}: {e}",
            cenario_path.display()
        )
    })?;

    warn_about_unsupported_bracos(&cenario);

    let (cfg, cfg_warnings) = build_config(&cenario);
    for w in &cfg_warnings {
        eprintln!("[aviso] {w}");
    }

    let (worlds, canonical_nomes_f) = load_maps(&mapa_path)?;
    if canonical_nomes_f.len() != crate::NP {
        return Err(format!(
            "map trace's nomes_f has {} names, this build needs NP={} -- refusing to run \
             (SPEC 4.2: a mismatched sense layout must not be aligned by position)",
            canonical_nomes_f.len(),
            crate::NP
        ));
    }
    warn_about_side_mismatch(&cenario, &worlds);

    let protocol = read_protocol(&cenario, &raw)?;
    let (campeao_w, campeao_nomes_f) = load_campeao_map(&campeao_path)?;

    eprintln!(
        "== taxi_evolve | cenario={} mapa={} (NC={}) campeao={} | P={} G={} R={} paciencia={:?} sigma_mut={} \
         crn={} campeao_mode={} seeds={:?} T={} ==",
        cenario_path.display(),
        mapa_path.display(),
        worlds.len(),
        campeao_path.display(),
        protocol.pop,
        protocol.generations,
        protocol.replicas,
        protocol.patience,
        protocol.sigma_mut,
        protocol.crn_str,
        protocol.campeao_str,
        protocol.seeds,
        protocol.ticks,
    );

    let mut sim = SimCpu::new();
    let champion_mode = if protocol.campeao_str == "reavaliado" {
        ChampionMode::Reevaluated
    } else {
        ChampionMode::Last
    };
    let crn = protocol.crn_str == "fixo";

    let mut all_measures: BTreeMap<u64, Measures> = BTreeMap::new();
    let mut all_champions: BTreeMap<u64, Champion> = BTreeMap::new();

    for &sd in &protocol.seeds {
        let champ_genome = champion_for_seed(
            &campeao_w,
            &campeao_nomes_f,
            &canonical_nomes_f,
            sd,
            &campeao_path,
        )?;
        let ga_params = GaParams {
            pop: protocol.pop,
            generations: protocol.generations,
            patience: protocol.patience,
            replicas: protocol.replicas,
            seeds: vec![sd],
            sigma_mut: protocol.sigma_mut,
            seed_from: Some(champ_genome),
            champion_mode,
            crn,
            ticks: protocol.ticks,
            ..GaParams::default()
        };

        let t0 = std::time::Instant::now();
        let mut champs = evolve(&mut sim, &worlds, &cfg, &ga_params);
        let champ = champs
            .remove(&sd)
            .expect("evolve must return exactly the one seed it was given");
        let measures = judge(&mut sim, &champ.genome, &worlds, &cfg, protocol.ticks);
        eprintln!(
            "  [semente {sd}] fit={:.3} ger_efetiva={} mortos={:.2} ent={:.2} ({:.1}s)",
            measures.fit,
            champ.hist.len(),
            measures.deaths,
            measures.deliveries,
            t0.elapsed().as_secs_f32(),
        );
        all_champions.insert(sd, champ);
        all_measures.insert(sd, measures);
    }

    let report_params = GaParams {
        seeds: protocol.seeds.clone(),
        sigma_mut: protocol.sigma_mut,
        ..GaParams::default()
    };
    let braco_value = measures_to_json(&all_measures, &all_champions, &report_params);
    let mut braco_obj = braco_value
        .as_object()
        .cloned()
        .expect("measures_to_json must return a JSON object");
    // `measures_to_json` (judge.rs) hardcodes "rotativo"/"ultimo" -- it predates
    // T7's `GaParams::crn`/`champion_mode` and this task's edit list does not
    // include judge.rs. Patch the two fields here, post-hoc, to the values this
    // run ACTUALLY used, plus a `motor` tag distinguishing a Rust-CPU run from a
    // Python/GPU one in a directory `analise_campo.py --mais` might combine.
    braco_obj.insert("crn".to_string(), json!(protocol.crn_str));
    braco_obj.insert("campeao".to_string(), json!(protocol.campeao_str));
    braco_obj.insert("motor".to_string(), json!("rust-cpu"));

    let mut bracos = serde_json::Map::new();
    bracos.insert("linear".to_string(), Value::Object(braco_obj));

    let mut out = serde_json::Map::new();
    out.insert("nomes_f".to_string(), json!(canonical_nomes_f));
    out.insert(
        "cenario".to_string(),
        json!(cenario_path.display().to_string()),
    );
    out.insert("bracos".to_string(), Value::Object(bracos));

    if let Some(parent) = saida_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "could not create --saida directory {}: {e}",
                    parent.display()
                )
            })?;
        }
    }
    let text = serde_json::to_string_pretty(&Value::Object(out))
        .map_err(|e| format!("could not serialize output JSON: {e}"))?;
    fs::write(&saida_path, text)
        .map_err(|e| format!("could not write --saida {}: {e}", saida_path.display()))?;

    Ok(saida_path)
}

// --- cenario -> Config ------------------------------------------------------

/// Maps `cenario["mundo"|"passageiros"|"corpo"]` onto `Config` fields that have
/// a direct equivalent (see `constants::Config`'s own field list). Anything the
/// cenario asks for that this crate structurally cannot honour (a different
/// `NPX`, `custo_gene != 0`, a non-`'vida'` ruler, `pax_cidade`, `ponto != 0`,
/// `jardim_comum`) is reported as a warning STRING, never silently dropped —
/// this task's own doctrine ("declare the negative with the mechanism").
fn build_config(cenario: &Value) -> (Config, Vec<String>) {
    let mut cfg = Config::default();
    let mut warnings = Vec::new();

    if let Some(mundo) = cenario.get("mundo") {
        if let Some(regua) = mundo.get("regua").and_then(Value::as_str) {
            if regua != "vida" {
                warnings.push(format!(
                    "mundo.regua = '{regua}' is not supported -- this crate implements ONLY the \
                     'vida' ruler (judge.rs's own doc comment: \"only the life ruler\"); running with 'vida' regardless"
                ));
            }
        }
        if let Some(precos) = mundo.get("precos") {
            if let Some(v) = precos.get("amp").and_then(Value::as_f64) {
                cfg.day_price_amplitude = v as f32;
            }
            if let Some(v) = precos.get("periodo").and_then(Value::as_f64) {
                cfg.day_price_period = v as f32;
            }
        }
    }

    if let Some(px) = cenario.get("passageiros") {
        if let Some(npx) = px.get("npx").and_then(Value::as_u64) {
            if npx as usize != crate::NPX {
                warnings.push(format!(
                    "passageiros.npx = {npx} is not supported -- NPX = {} is a compile-time \
                     constant in this crate (constants.rs); ignoring",
                    crate::NPX
                ));
            }
        }
        if let Some(v) = px.get("respawn").and_then(Value::as_f64) {
            cfg.respawn_denominator = v as f32;
        }
        if let Some(v) = px.get("tarifa_base").and_then(Value::as_f64) {
            cfg.fare_base = v as f32;
        }
        if let Some(v) = px.get("tarifa_km").and_then(Value::as_f64) {
            cfg.fare_per_distance = v as f32;
        }
        if px.get("pax_cidade").and_then(Value::as_bool) == Some(true) {
            warnings.push("passageiros.pax_cidade = true is not supported in v0 (SPEC 12: out of scope); ignoring".to_string());
        }
        if px
            .get("ponto")
            .and_then(Value::as_f64)
            .is_some_and(|v| v != 0.0)
        {
            warnings.push(
                "passageiros.ponto != 0 is not supported in v0 (SPEC 7.5: not implemented, \
                 consumes RNG and breaks determinism); ignoring"
                    .to_string(),
            );
        }
    }

    if let Some(corpo) = cenario.get("corpo") {
        if let Some(v) = corpo.get("dinh0").and_then(Value::as_f64) {
            cfg.initial_debt = v as f32;
        }
        if let Some(v) = corpo.get("fome_passo").and_then(Value::as_f64) {
            cfg.hunger_per_tick = v as f32;
        }
        if let Some(v) = corpo.get("kcal_pane").and_then(Value::as_f64) {
            cfg.breakdown_kcal = v as f32;
        }
        if let Some(v) = corpo.get("guincho").and_then(Value::as_f64) {
            cfg.tow_cost = v as f32;
        }
        if let Some(v) = corpo.get("fuel_pos_pane").and_then(Value::as_f64) {
            cfg.breakdown_fuel_refill = v as f32;
        }
        if corpo
            .get("custo_gene")
            .and_then(Value::as_f64)
            .is_some_and(|v| v != 0.0)
        {
            warnings.push(
                "corpo.custo_gene != 0 is not supported in v0 (SPEC: gene_cost is always 0, \
                 the Linear brain has dof=0 -- judge.rs's DOF_V0/METABOLISM_V0); ignoring"
                    .to_string(),
            );
        }
    }

    if cenario
        .get("protocolo")
        .and_then(|p| p.get("jardim_comum"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        warnings.push(
            "protocolo.jardim_comum is not a togglable flag in this v0 CPU port -- SimCpu's \
             birth sampling already shares one cradle per (map, replica) across every genome of \
             a generation by construction (sim_cpu.rs's module doc); ignoring the flag"
                .to_string(),
        );
    }

    (cfg, warnings)
}

fn warn_about_unsupported_bracos(cenario: &Value) {
    let Some(bracos) = cenario.get("bracos").and_then(Value::as_array) else {
        return;
    };
    for b in bracos {
        let name = b
            .as_str()
            .or_else(|| b.get("rotulo").and_then(Value::as_str))
            .or_else(|| b.get("nome").and_then(Value::as_str))
            .unwrap_or("<unnamed>");
        if name != "linear" {
            eprintln!(
                "[warning] ignoring arm '{name}': this v0 CPU port implements only the 'linear' \
                 brain (no non-Linear brain exists in this crate yet)"
            );
        }
    }
}

fn warn_about_side_mismatch(cenario: &Value, worlds: &[World]) {
    let Some(l) = cenario
        .get("mundo")
        .and_then(|m| m.get("L"))
        .and_then(Value::as_f64)
    else {
        return;
    };
    for (i, w) in worlds.iter().enumerate() {
        if (w.side - l as f32).abs() > 1e-3 {
            eprintln!(
                "[aviso] cenario mundo.L={l} does not match map {i}'s side={} -- v0 never \
                 generates the map (SPEC 2.2), the trace's own L is authoritative",
                w.side
            );
        }
    }
}

// --- protocolo ---------------------------------------------------------------

struct Protocol {
    pop: usize,
    generations: usize,
    patience: Option<usize>,
    replicas: usize,
    seeds: Vec<u64>,
    sigma_mut: f32,
    campeao_str: String,
    crn_str: String,
    ticks: usize,
}

fn read_protocol(cenario: &Value, raw: &RawArgs) -> Result<Protocol, String> {
    let protocolo = cenario.get("protocolo");
    let field_u64 = |name: &str| protocolo.and_then(|p| p.get(name)).and_then(Value::as_u64);
    let field_f64 = |name: &str| protocolo.and_then(|p| p.get(name)).and_then(Value::as_f64);
    let field_str = |name: &str| protocolo.and_then(|p| p.get(name)).and_then(Value::as_str);

    let pop = raw
        .one_usize("P")?
        .unwrap_or_else(|| field_u64("P").unwrap_or(128) as usize);
    let generations = raw
        .one_usize("G")?
        .unwrap_or_else(|| field_u64("G").unwrap_or(30) as usize);
    let replicas = raw
        .one_usize("R")?
        .unwrap_or_else(|| field_u64("R").unwrap_or(4) as usize);
    let sigma_mut = raw
        .one_f32("sigma-mut")?
        .unwrap_or_else(|| field_f64("sigma_mut").unwrap_or(0.1) as f32);
    let patience = match raw.one_usize("paciencia")? {
        Some(v) => Some(v),
        None => field_u64("paciencia").map(|v| v as usize),
    };
    let campeao_str = field_str("campeao").unwrap_or("ultimo").to_string();
    let crn_str = field_str("crn").unwrap_or("rotativo").to_string();

    let seeds: Vec<u64> = match raw.many("seeds") {
        Some(vs) => vs
            .iter()
            .map(|s| {
                s.parse::<u64>()
                    .map_err(|_| format!("--seeds: '{s}' is not a valid seed number"))
            })
            .collect::<Result<_, _>>()?,
        None => protocolo
            .and_then(|p| p.get("seeds"))
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_else(|| vec![0, 1, 2, 3, 4]),
    };
    if seeds.is_empty() {
        return Err("protocolo.seeds (or --seeds) resolved to an empty list".to_string());
    }

    let ticks = ticks_from_mundo(cenario.get("mundo"));

    Ok(Protocol {
        pop,
        generations,
        patience,
        replicas,
        seeds,
        sigma_mut,
        campeao_str,
        crn_str,
        ticks,
    })
}

/// SPEC §10 `T_evo = T_aval`, `e43_campo.py`'s own rule: `horizonte='aberto'`
/// uses `mundo.T_max` (a hard ceiling, `corte_vivos` stops earlier once
/// everyone is dead); `horizonte='fixo'` (or missing) uses `240 * mundo.dias`.
fn ticks_from_mundo(mundo: Option<&Value>) -> usize {
    let dias = mundo
        .and_then(|m| m.get("dias"))
        .and_then(Value::as_f64)
        .unwrap_or(10.0);
    let horizonte = mundo
        .and_then(|m| m.get("horizonte"))
        .and_then(Value::as_str)
        .unwrap_or("fixo");
    if horizonte == "aberto" {
        mundo
            .and_then(|m| m.get("T_max"))
            .and_then(Value::as_u64)
            .unwrap_or(240 * 30) as usize
    } else {
        (240.0 * dias).round() as usize
    }
}

// --- map loading ---------------------------------------------------------

/// `--mapa`: a single trace JSON, or a directory of them (`NC` maps). Returns
/// the loaded `World`s plus the FIRST trace's `nomes_f` (the canonical gene
/// order every champion genome is aligned against, `champion_for_seed`).
fn load_maps(path: &Path) -> Result<(Vec<World>, Vec<String>), String> {
    let mut files: Vec<PathBuf> = if path.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| format!("could not read --mapa directory {}: {e}", path.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
            .collect();
        entries.sort();
        entries
    } else {
        vec![path.to_path_buf()]
    };
    if files.is_empty() {
        return Err(format!(
            "--mapa {} has no .json trace files",
            path.display()
        ));
    }
    files.sort();

    let mut worlds = Vec::with_capacity(files.len());
    let mut canonical_nomes_f: Option<Vec<String>> = None;
    for f in &files {
        let trace =
            Trace::load(f).map_err(|e| format!("could not load map trace {}: {e}", f.display()))?;
        if canonical_nomes_f.is_none() {
            canonical_nomes_f = Some(trace.nomes_f.clone());
        }
        worlds.push(trace.to_world());
    }
    Ok((
        worlds,
        canonical_nomes_f.expect("at least one file was loaded"),
    ))
}

// --- campeao loading + gene-name alignment --------------------------------

/// `{seed: 40 floats}`, one entry per seed the champion file's
/// `bracos.linear.W` carries.
type CampeaoWeights = HashMap<u64, Vec<f32>>;

/// `--campeao`'s `bracos.linear.W`, `{seed: 40 floats}`, plus the file's OWN
/// `nomes_f` if it declares one (some older `e43_*.json` files predate that
/// key).
fn load_campeao_map(path: &Path) -> Result<(CampeaoWeights, Option<Vec<String>>), String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("could not read --campeao {}: {e}", path.display()))?;
    let doc: Value = serde_json::from_str(&text)
        .map_err(|e| format!("could not parse --campeao JSON {}: {e}", path.display()))?;

    let nomes_f = doc.get("nomes_f").and_then(Value::as_array).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect::<Vec<String>>()
    });

    let w = doc
        .pointer("/bracos/linear/W")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!(
                "--campeao {} has no bracos.linear.W (only the 'linear' arm is supported in v0)",
                path.display()
            )
        })?;

    let mut map = HashMap::new();
    for (key, value) in w {
        let seed: u64 = key
            .parse()
            .map_err(|_| format!("--campeao bracos.linear.W key '{key}' is not a seed number"))?;
        let genome: Vec<f32> = value
            .as_array()
            .ok_or_else(|| format!("--campeao bracos.linear.W['{key}'] is not an array"))?
            .iter()
            .map(|x| {
                x.as_f64().map(|f| f as f32).ok_or_else(|| {
                    format!("--campeao bracos.linear.W['{key}'] has a non-numeric weight")
                })
            })
            .collect::<Result<_, _>>()?;
        map.insert(seed, genome);
    }
    if map.is_empty() {
        return Err(format!(
            "--campeao {} has an empty bracos.linear.W",
            path.display()
        ));
    }
    Ok((map, nomes_f))
}

/// Resolves seed `sd`'s champion genome: exact seed match, else a cyclic
/// fallback over the file's OWN sorted seeds (mirrors `e43_campo.py`'s
/// `alinha_semente`/`--init-from` fallback: `J.get(str(sd), J[sorted(J)[sd %
/// len(J)]])` -- a champion file with fewer seeds than requested still seeds
/// every requested seed, deterministically). Realigns by NAME when the source
/// file's own `nomes_f` differs from `canonical` (SPEC 4.3's `alinha_semente`);
/// otherwise trusts the array is already in `canonical` order. Refuses (per
/// this task's own rule) when the aligned genome is not exactly `NP` weights.
fn champion_for_seed(
    w_map: &HashMap<u64, Vec<f32>>,
    src_nomes_f: &Option<Vec<String>>,
    canonical: &[String],
    seed: u64,
    campeao_path: &Path,
) -> Result<Genome, String> {
    let raw: &Vec<f32> = if let Some(v) = w_map.get(&seed) {
        v
    } else {
        let mut keys: Vec<u64> = w_map.keys().copied().collect();
        keys.sort_unstable();
        let fallback = keys[(seed as usize) % keys.len()];
        eprintln!(
            "[aviso] --campeao {} has no seed {seed}; using seed {fallback}'s champion instead \
             (cyclic fallback, mirrors e43_campo.py's --init-from behaviour)",
            campeao_path.display()
        );
        w_map
            .get(&fallback)
            .expect("fallback key came from w_map's own keys")
    };

    let aligned: Vec<f32> = match src_nomes_f {
        Some(src) if src.as_slice() != canonical => {
            if src.len() != raw.len() {
                return Err(format!(
                    "--campeao seed {seed}: nomes_f has {} names but W has {} weights -- cannot align",
                    src.len(),
                    raw.len()
                ));
            }
            let pos: HashMap<&str, usize> = src.iter().map(String::as_str).zip(0..).collect();
            canonical
                .iter()
                .map(|name| pos.get(name.as_str()).map(|&i| raw[i]).unwrap_or(0.0))
                .collect()
        }
        _ => raw.clone(),
    };

    if aligned.len() != crate::NP {
        return Err(format!(
            "--campeao seed {seed}: genome has {} weights after alignment, this build needs NP={} -- refusing to run",
            aligned.len(),
            crate::NP
        ));
    }
    Ok(Genome(aligned))
}

// --- tiny manual CLI parser (no new dependency) ---------------------------

/// `--flag value1 value2 ...` -> `{"flag": ["value1", "value2", ...]}`. No new
/// crate (this task's hard rule) -- `clap` is not an option.
struct RawArgs(HashMap<String, Vec<String>>);

impl RawArgs {
    fn parse(args: &[String]) -> Result<RawArgs, String> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            let Some(flag) = arg.strip_prefix("--") else {
                return Err(format!(
                    "unexpected positional argument '{arg}' (every argument must be --flag value...)"
                ));
            };
            let mut values = Vec::new();
            let mut j = i + 1;
            while j < args.len() && !args[j].starts_with("--") {
                values.push(args[j].clone());
                j += 1;
            }
            map.insert(flag.to_string(), values);
            i = j;
        }
        Ok(RawArgs(map))
    }

    fn one(&self, name: &str) -> Option<&str> {
        self.0.get(name).and_then(|v| v.first()).map(String::as_str)
    }

    fn many(&self, name: &str) -> Option<&[String]> {
        self.0.get(name).map(Vec::as_slice)
    }

    fn required_path(&self, name: &str) -> Result<PathBuf, String> {
        self.one(name)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing required --{name}"))
    }

    fn one_usize(&self, name: &str) -> Result<Option<usize>, String> {
        match self.one(name) {
            None => Ok(None),
            Some(s) => s
                .parse::<usize>()
                .map(Some)
                .map_err(|_| format!("--{name}: '{s}' is not a valid integer")),
        }
    }

    fn one_f32(&self, name: &str) -> Result<Option<f32>, String> {
        match self.one(name) {
            None => Ok(None),
            Some(s) => s
                .parse::<f32>()
                .map(Some)
                .map_err(|_| format!("--{name}: '{s}' is not a valid number")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_args_parses_repeated_multi_value_flags_and_single_value_flags() {
        let args: Vec<String> = ["--seeds", "0", "1", "2", "--P", "8", "--tag", "x"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let raw = RawArgs::parse(&args).unwrap();
        assert_eq!(
            raw.many("seeds"),
            Some(["0".to_string(), "1".to_string(), "2".to_string()].as_slice())
        );
        assert_eq!(raw.one("P"), Some("8"));
        assert_eq!(raw.one("tag"), Some("x"));
        assert_eq!(raw.one("missing"), None);
    }

    #[test]
    fn ticks_from_mundo_uses_t_max_only_when_horizonte_is_aberto() {
        let aberto = serde_json::json!({"horizonte": "aberto", "T_max": 7200, "dias": 10});
        assert_eq!(ticks_from_mundo(Some(&aberto)), 7200);
        let fixo = serde_json::json!({"horizonte": "fixo", "dias": 5});
        assert_eq!(ticks_from_mundo(Some(&fixo)), 1200);
        assert_eq!(
            ticks_from_mundo(None),
            2400,
            "default: fixo, dias=10 -> 2400 ticks"
        );
    }

    #[test]
    fn champion_for_seed_refuses_a_genome_that_is_not_np_weights_long() {
        let mut w = HashMap::new();
        w.insert(0u64, vec![0.0f32; 39]);
        let canonical: Vec<String> = (0..40).map(|i| format!("f{i}")).collect();
        let err = champion_for_seed(&w, &None, &canonical, 0, Path::new("dummy.json")).unwrap_err();
        assert!(err.contains("NP=40"), "got: {err}");
    }

    #[test]
    fn champion_for_seed_falls_back_cyclically_to_another_seed() {
        let mut w = HashMap::new();
        w.insert(3u64, vec![1.0f32; 40]);
        let canonical: Vec<String> = (0..40).map(|i| format!("f{i}")).collect();
        let genome = champion_for_seed(&w, &None, &canonical, 99, Path::new("dummy.json")).unwrap();
        assert_eq!(genome.0, vec![1.0f32; 40]);
    }

    #[test]
    fn champion_for_seed_realigns_by_name_when_nomes_f_differs() {
        // canonical = [f0, f1, ..., f39]; source = canonical REVERSED, with
        // raw[k] = k as f32 -- so aligned[i] (name f{i}) must come from
        // raw[39 - i] (f{i}'s position in the reversed source order).
        let canonical: Vec<String> = (0..40).map(|i| format!("f{i}")).collect();
        let src_nomes_f: Vec<String> = canonical.iter().rev().cloned().collect();
        let raw: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let mut w = HashMap::new();
        w.insert(0u64, raw.clone());

        let genome = champion_for_seed(
            &w,
            &Some(src_nomes_f),
            &canonical,
            0,
            Path::new("dummy.json"),
        )
        .unwrap();
        for i in 0..40 {
            assert_eq!(
                genome.0[i],
                raw[39 - i],
                "gene {i} ({}) misaligned",
                canonical[i]
            );
        }
    }

    #[test]
    fn champion_for_seed_a_name_missing_from_the_source_becomes_zero() {
        let canonical: Vec<String> = vec!["a".to_string(), "b".to_string()]
            .into_iter()
            .chain((2..40).map(|i| format!("f{i}")))
            .collect();
        // source has every name EXCEPT "b" (missing on purpose), same length as
        // canonical so the length guard does not fire first.
        let mut src_nomes_f = canonical.clone();
        src_nomes_f[1] = "not_in_canonical".to_string();
        let mut raw = vec![0.0f32; 40];
        raw[1] = 999.0; // sits at "not_in_canonical"'s slot, must never surface
        let mut w = HashMap::new();
        w.insert(0u64, raw);

        let genome = champion_for_seed(
            &w,
            &Some(src_nomes_f),
            &canonical,
            0,
            Path::new("dummy.json"),
        )
        .unwrap();
        assert_eq!(
            genome.0[1], 0.0,
            "a canonical name absent from the source must default to 0.0, not leak an unrelated weight"
        );
    }
}
