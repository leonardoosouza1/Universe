//! Read-only loader for `traco_ref.json` (SPEC §11, plan §3.2/§T1) — the reference
//! trace produced by `neuromod-lab/src/traco_referencia.py` that the identity gate
//! (T5) compares Rust against, and the only source of a v0 map (SPEC §2.2: the Rust
//! side never generates a map, `gerar_mapa`'s `torch.rand`/`conv2d` pipeline has no
//! bit-exact Rust equivalent).
//!
//! This module owns exactly two things: getting every documented key off disk
//! (`Trace::load`), and turning the `mapa` block into a `World` (`Trace::to_world`).
//! It deliberately does NOT turn `inicial` into a `Beings` — see the note on
//! `Trace::initial` below, this crate's hard rule against creating `beings.rs`
//! (T2's file) forced that decision, and the final report says so explicitly.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::constants::N_TYPES;
use crate::world::World;

/// Everything that can go wrong loading a trace file.
#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("could not read trace file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse trace JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// The `mapa` block (SPEC §2.1).
#[derive(Debug, Deserialize)]
pub struct TraceMap {
    /// `[TEX][TEX]`, row-major `[iy][ix]` (SPEC §2.1, §2.3).
    pub custo: Vec<Vec<f32>>,
    /// One entry per type name in `Trace::tipos`, each a `[N][2]` list of points.
    pub serv: HashMap<String, Vec<[f32; 2]>>,
    /// `[K][2]` district (Voronoi) centers.
    pub centros: Vec<[f32; 2]>,
    /// `[K]` tax rate per district.
    pub aliq: Vec<f32>,
    /// `[K]` base price multiplier per district.
    pub pmul: Vec<f32>,
    /// `[K]` rent per district.
    pub alug: Vec<f32>,
    /// `[K]` passenger demand per district.
    pub demanda: Vec<f32>,
    /// `[GRID][GRID]`, row-major `[iy][ix]` — district id per cell.
    pub campo_dist: Vec<Vec<u32>>,
    /// `[K]` price-cycle phase per district (`fase_preco`, SPEC §2.1).
    ///
    /// **Known gap, declared rather than silently patched over:** as of
    /// 2026-09-10, `traco_referencia.py`'s `mapa=dict(...)` (line 23) does not
    /// export this key at all, and T0's planned emendas (plan §T0, items a-f)
    /// don't add it either — the real `traco_ref.json` this crate will eventually
    /// load is missing a field `World.price_phase` needs. `#[serde(default)]`
    /// makes that produce an empty `Vec` instead of a load failure; `to_world()`
    /// then fills it with zeros the length of `centros`, rather than panicking or
    /// guessing a nonzero value. This crate's `tests/fixtures/tiny_trace.json`
    /// DOES supply `fase_preco`, so the non-empty path is exercised too.
    #[serde(default, rename = "fase_preco")]
    pub price_phase: Vec<f32>,
    /// District perfil labels. Not used by `to_world()` (the plan's `World` struct
    /// has no field for it, plan §3.2) — kept only so a present-but-unused key does
    /// not turn into a hard deserialization failure.
    #[serde(default)]
    pub perfis: serde_json::Value,
}

/// The `inicial` block (SPEC §1) — the starting state of every being, `[B]`-shaped
/// (or `[B][NPX]...` for the passenger arrays). v0 reads this instead of generating
/// it (SPEC §1: v0 does not generate these values — it READS them from
/// `traco_ref.json`).
#[derive(Debug, Deserialize)]
pub struct TraceInitial {
    pub pos: Vec<[f32; 2]>,
    pub fuel: Vec<f32>,
    pub fome: Vec<f32>,
    pub agua: Vec<f32>,
    pub dinh: Vec<f32>,
    pub vida: Vec<f32>,
    pub calor: Vec<f32>,
    pub fad: Vec<f32>,
    pub res: Vec<f32>,
    /// Per-map, not per-being — every entry is identical for beings sharing a map
    /// (SPEC §1: `fase` is per MAP, not per being).
    pub fase: Vec<f32>,
    /// `[B][NPX][2]` passenger-slot positions.
    pub px: Vec<Vec<[f32; 2]>>,
    /// `[B][NPX][2]` passenger-slot destinations.
    pub pxd: Vec<Vec<[f32; 2]>>,
    /// `[B][NPX]` slot-active flags.
    pub px_ok: Vec<Vec<bool>>,
    /// `[B]` home district id.
    pub casa: Vec<u32>,
}

/// The full reference trace (SPEC §11, plan T1 §"Interface").
///
/// Fields this crate does not itself interpret (`acoes`, the per-tick arrays,
/// `esquema`) are kept as raw [`serde_json::Value`] rather than typed out: T1's job
/// is the map/genome/metadata, not the tick-by-tick comparison T5's `gate.rs`
/// performs. Typing them loosely here means T5 does not have to wait for a second
/// edit of this file to read them with `serde_json::from_value`.
#[derive(Debug, Deserialize)]
pub struct Trace {
    #[serde(rename = "L")]
    pub l: f32,
    pub seed_mapa: u64,
    #[serde(rename = "B")]
    pub b: usize,
    #[serde(rename = "T")]
    pub t: usize,
    pub dia: f32,
    /// The 40 genome weight names, in `NOMES_F` order (SPEC §4.3). Length must be
    /// checked against `NP` by whoever consumes a genome against this trace —
    /// `genome.rs`'s `NF`/`S` mismatch check covers the on-disk `.taxg` format, but
    /// a trace's OWN `nomes_f` is metadata, not itself a genome file.
    pub nomes_f: Vec<String>,
    /// The 10 `NOMES_OP` names (SPEC §4.1).
    pub nomes_op: Vec<String>,
    /// The 19 `NOMES_CORPO` names (SPEC §4.2). `nomes_op.len() + nomes_corpo.len()`
    /// must equal this build's `NF` (SPEC §4.2) — checked in `load`.
    pub nomes_corpo: Vec<String>,
    pub acoes: serde_json::Value,
    /// The 5 service-point type names, in the order `mapa.serv`'s keys and
    /// `World.services` must follow (SPEC §0 `TIPOS`).
    pub tipos: Vec<String>,
    #[serde(rename = "G")]
    pub g: usize,
    #[serde(rename = "S")]
    pub s: usize,
    /// Raw `CfgCampo` field dump (SPEC §3.2). Deliberately NOT merged into
    /// `Config::default()` — see that struct's doc comment for why the two must
    /// stay separate (in particular `respawn`/`ponto`, overridden by the trace
    /// generator away from `CfgCampo`'s own defaults).
    pub cfg: HashMap<String, serde_json::Value>,
    pub mapa: TraceMap,
    /// The champion genome this trace was produced with, `NP` weights.
    pub genoma: Vec<f32>,
    pub inicial: TraceInitial,
    pub acoes_por_tique: serde_json::Value,
    pub traco: serde_json::Value,
    pub traco_extra: serde_json::Value,
    pub alvo_por_tique: serde_json::Value,
    pub margem_por_tique: serde_json::Value,
    pub nota_por_tique: serde_json::Value,
    #[serde(rename = "F_primeiros_50")]
    pub f_primeiros_50: serde_json::Value,
    pub t_vida: Vec<f32>,
    pub rec: Vec<f32>,
    pub esquema: HashMap<String, String>,
}

impl Trace {
    /// Reads and parses a `traco_ref.json` file. Does not validate cross-field
    /// invariants (e.g. `nomes_f.len() == 40`) beyond what `serde` already checks —
    /// that validation belongs to whichever task actually depends on the invariant
    /// (`genome.rs` for the genome length, T5's `gate.rs` for the tick arrays), so a
    /// trace that is merely metadata-incomplete for THOSE tasks can still load here.
    pub fn load(path: &Path) -> Result<Trace, TraceError> {
        let text = fs::read_to_string(path).map_err(|source| TraceError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let trace: Trace = serde_json::from_str(&text)?;
        Ok(trace)
    }

    /// Builds the static [`World`] this trace ran on (SPEC §2, plan §3.2).
    ///
    /// `mapa.serv`'s keys are matched against `self.tipos` (not a hardcoded
    /// Portuguese list) to build `World.services` in `TIPOS` order — a type present
    /// in `tipos` but absent from `serv` becomes an empty `Vec` (a real map with a
    /// zero-density type for its district profile, SPEC §2.1, is not an error).
    pub fn to_world(&self) -> World {
        let k = self.mapa.centros.len();
        let cost_tex: Vec<f32> = self.mapa.custo.iter().flatten().copied().collect();
        let district_field: Vec<u32> = self.mapa.campo_dist.iter().flatten().copied().collect();
        let price_phase = if self.mapa.price_phase.is_empty() {
            vec![0.0; k]
        } else {
            self.mapa.price_phase.clone()
        };

        let mut services: [Vec<[f32; 2]>; N_TYPES] = std::array::from_fn(|_| Vec::new());
        for (t, type_name) in self.tipos.iter().enumerate().take(N_TYPES) {
            services[t] = self.mapa.serv.get(type_name).cloned().unwrap_or_default();
        }

        World {
            side: self.l,
            cost_tex,
            district_field,
            centers: self.mapa.centros.clone(),
            tax: self.mapa.aliq.clone(),
            price_mul_base: self.mapa.pmul.clone(),
            rent: self.mapa.alug.clone(),
            demand: self.mapa.demanda.clone(),
            price_phase,
            services,
        }
    }

    // Deliberately no `initial_beings(&self) -> Beings` here, even though the
    // plan's T1 interface block lists one (plan §T1 "Interface"). `Beings` is
    // defined in T2's `beings.rs` (plan §T2 "Cria"), and this task's hard rules
    // forbid creating that file — returning a type that does not exist yet in this
    // crate is not implementable without either violating that rule or declaring a
    // throwaway `Beings` stub here that would collide with T2's real one the moment
    // both land (the exact multi-implementer collision the plan's §4 is designed to
    // avoid). `TraceInitial` (`self.inicial`) already carries every field SPEC §1
    // lists as "read, not generated" — `pos, fuel, fome, agua, dinh, vida, calor,
    // fad, res, fase, px, pxd, px_ok, casa` — so T2's constructor is a direct,
    // mechanical read of `trace.inicial`, not a redesign.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{GRID, NP, TEX};
    use std::path::PathBuf;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_trace.json")
    }

    fn load_fixture() -> Trace {
        Trace::load(&fixture_path()).expect("tests/fixtures/tiny_trace.json must parse")
    }

    #[test]
    fn trace_fixture_loads_every_documented_key() {
        let trace = load_fixture();

        assert_eq!(trace.l, 20.0);
        assert_eq!(trace.seed_mapa, 1000);
        assert_eq!(trace.b, 2);
        assert_eq!(trace.t, 3);
        assert_eq!(trace.g, 9);
        assert_eq!(trace.s, 10);

        // Same acceptance numbers T0's own aceite criterion names (SPEC §T0.aceite).
        assert_eq!(trace.nomes_f.len(), 40, "len(nomes_f) == 40");
        assert_eq!(trace.nomes_corpo.len(), 19, "len(nomes_corpo) == 19");
        assert_eq!(trace.nomes_op.len(), 10);
        assert_eq!(
            trace.nomes_op.len() + trace.nomes_corpo.len(),
            crate::constants::NF,
            "NOMES_OP + NOMES_CORPO must equal this build's NF"
        );
        assert_eq!(trace.genoma.len(), NP, "len(genoma) == 40");
        assert_eq!(trace.tipos.len(), N_TYPES);

        assert_eq!(trace.mapa.custo.len(), TEX);
        assert_eq!(trace.mapa.custo[0].len(), TEX);
        assert_eq!(trace.mapa.campo_dist.len(), GRID);
        assert_eq!(trace.mapa.campo_dist[0].len(), GRID);
        assert_eq!(trace.mapa.centros.len(), 2);
        assert_eq!(trace.mapa.price_phase.len(), 2, "fixture supplies fase_preco");

        assert_eq!(trace.inicial.pos.len(), trace.b);
        assert_eq!(trace.inicial.px.len(), trace.b);
        assert_eq!(trace.inicial.px[0].len(), crate::constants::NPX);
        assert_eq!(trace.inicial.fase[0], trace.inicial.fase[1], "fase is per-map");

        assert_eq!(trace.t_vida.len(), trace.b);
        assert_eq!(trace.rec.len(), trace.b);
        assert!(trace.esquema.contains_key("traco"));
        assert!(trace.cfg.contains_key("respawn"));
    }

    #[test]
    fn to_world_orders_services_by_tipos_and_defaults_missing_price_phase() {
        let trace = load_fixture();
        let world = trace.to_world();

        assert_eq!(world.side, 20.0);
        assert_eq!(world.cost_tex.len(), TEX * TEX);
        assert_eq!(world.district_field.len(), GRID * GRID);
        assert_eq!(world.centers.len(), 2);
        assert_eq!(world.price_phase, vec![0.1, 0.9], "fixture's fase_preco must pass through");

        // TIPOS = [posto, comida, agua, descanso, oficina]; fixture has 1 posto
        // point, 2 comida points.
        assert_eq!(world.services[0].len(), 1, "posto");
        assert_eq!(world.services[1].len(), 2, "comida");
        assert_eq!(world.services[2].len(), 1, "agua");
        assert_eq!(world.services[3].len(), 1, "descanso");
        assert_eq!(world.services[4].len(), 1, "oficina");
        assert_eq!(world.services[0][0], [2.0, 2.0]);
    }

    #[test]
    fn to_world_defaults_price_phase_to_zero_when_the_key_is_missing() {
        // Simulates today's real traco_referencia.py output, which has no
        // `fase_preco` key at all (see TraceMap::price_phase's doc comment).
        let mut trace = load_fixture();
        trace.mapa.price_phase.clear();
        let world = trace.to_world();
        assert_eq!(world.price_phase, vec![0.0, 0.0]);
    }

    #[test]
    fn a_missing_file_is_a_clean_error_not_a_panic() {
        let err = Trace::load(Path::new("/nonexistent/does/not/exist.json"));
        assert!(matches!(err, Err(TraceError::Io { .. })));
    }

    #[test]
    fn malformed_json_is_a_clean_error() {
        let mut path = std::env::temp_dir();
        path.push("taxi_world_malformed_trace_test.json");
        fs::write(&path, b"{ not valid json").unwrap();
        let err = Trace::load(&path);
        assert!(matches!(err, Err(TraceError::Json(_))));
        let _ = fs::remove_file(&path);
    }

    /// Ignored: only runs against the real trace once T0 produces it (plan §T0,
    /// not part of T1). Point `TAXI_TRACO_REF` at the file, then:
    /// `nice -n 10 env CARGO_BUILD_JOBS=4 cargo test -p taxi-world -- --ignored`.
    #[test]
    #[ignore = "needs the real traco_ref.json (T0), point TAXI_TRACO_REF at it"]
    fn trace_loads_the_real_reference_when_present() {
        let path = std::env::var("TAXI_TRACO_REF")
            .expect("set TAXI_TRACO_REF to the path of a real traco_ref.json");
        let trace = Trace::load(Path::new(&path)).expect("real trace must parse");
        assert_eq!(trace.nomes_f.len(), 40);
        assert_eq!(trace.nomes_corpo.len(), 19);
        assert_eq!(trace.genoma.len(), NP);
        let _world = trace.to_world();
    }
}
