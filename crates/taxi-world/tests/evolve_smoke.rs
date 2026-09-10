//! T7 acceptance test: runs `taxi_evolve` (via `taxi_world::taxi_evolve_cli::
//! run`, IN CODE, not as a subprocess — see that module's own doc comment for
//! why the CLI logic lives in the library) end to end against `tests/fixtures/
//! tiny_trace.json`, with P=8, G=2, R=1, seeds=[0], T=300, and checks the
//! output JSON carries every key `scripts/analise_campo.py`/`src/
//! retrato_bateria.py` (the lab, Python) read.
//!
//! A second, `#[ignore]`d test measures ONE generation's wall time at
//! P=128, R=4 (this crate's final report cites the number it prints) — kept
//! `#[ignore]`d and run manually with `--release` so the default `cargo test
//! -p taxi-world` suite stays fast (SPEC-scale, T=7200, in a debug build,
//! would not).

use std::fs;
use std::path::PathBuf;

use taxi_world::evolution::Simulador;
use taxi_world::sim_cpu::SimCpu;
use taxi_world::trace::Trace;
use taxi_world::{Config, Genome, NP};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_trace.json")
}

/// A private scratch directory for this test's INPUT fixtures (the cenario/
/// campeao JSON it writes for `taxi_evolve` to read) — `std::env::temp_dir()`
/// mirrors this crate's own precedent for test-only scratch files
/// (`trace.rs::tests::malformed_json_is_a_clean_error`). The run's real OUTPUT
/// goes through `--saida` to the storage-policy directory instead (see the
/// tests below), not here. Keyed by `tag`, NOT just the process id: `cargo
/// test` runs every test in this file as a THREAD of the same process, so two
/// tests sharing one process-id-only path would race on `fs::remove_dir_all`
/// at the other's expense.
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("taxi_evolve_smoke_{}_{tag}", std::process::id()));
    fs::create_dir_all(&dir).expect("must be able to create the scratch dir");
    dir
}

#[test]
fn taxi_evolve_runs_end_to_end_on_the_fixture_and_writes_the_e43_schema() {
    let trace = Trace::load(&fixture_path()).expect("fixture must load");
    let scratch = scratch_dir("smoke");

    let cenario_path = scratch.join("cenario.json");
    let cenario = serde_json::json!({
        "mundo": { "L": trace.l, "horizonte": "aberto", "T_max": 300, "regua": "vida" },
        "passageiros": { "npx": 12 },
        "corpo": {},
        "bracos": ["linear"],
        "protocolo": { "P": 8, "G": 2, "R": 1, "seeds": [0], "sigma_mut": 0.1 }
    });
    fs::write(&cenario_path, serde_json::to_string_pretty(&cenario).unwrap()).unwrap();

    let campeao_path = scratch.join("campeao.json");
    let campeao = serde_json::json!({
        "nomes_f": trace.nomes_f,
        "bracos": { "linear": { "W": { "0": trace.genoma } } },
    });
    fs::write(&campeao_path, serde_json::to_string_pretty(&campeao).unwrap()).unwrap();

    let saida_path = PathBuf::from("/media/llm-data/universe-runs/neuromod-lab-taxi/e82/e43_evolve_smoke.json");

    let args: Vec<String> = vec![
        "--cenario".into(),
        cenario_path.display().to_string(),
        "--mapa".into(),
        fixture_path().display().to_string(),
        "--campeao".into(),
        campeao_path.display().to_string(),
        "--saida".into(),
        saida_path.display().to_string(),
    ];

    let out_path = taxi_world::taxi_evolve_cli::run(&args).expect("taxi_evolve must succeed on the fixture");
    assert_eq!(out_path, saida_path, "run() must return the --saida path it wrote");

    let text = fs::read_to_string(&out_path).expect("output file must exist on disk");
    let json: serde_json::Value = serde_json::from_str(&text).expect("output must be valid JSON");

    assert!(json.get("nomes_f").is_some(), "top-level nomes_f missing");
    assert_eq!(json["nomes_f"].as_array().unwrap().len(), NP);
    assert!(json.get("cenario").is_some(), "top-level cenario missing");

    let linear = json.pointer("/bracos/linear").expect("bracos.linear must exist");
    for key in [
        "fit",
        "fit_por_semente",
        "mortos",
        "ent",
        "panes",
        "fome_fim",
        "agua_fim",
        "dinh",
        "res",
        "litros_por_abast",
        "n_posto",
        "n_come",
        "oficina",
        "rec",
        "t_vida",
        "t_vida_media",
        "frac_teto",
        "W",
        "ger_efetiva",
        "seeds",
        "sigma_mut",
        "crn",
        "campeao",
        "motor",
    ] {
        assert!(linear.get(key).is_some(), "missing key '{key}' in bracos.linear: {linear}");
    }
    assert_eq!(linear["motor"], "rust-cpu");
    assert_eq!(linear["seeds"], serde_json::json!([0]));
    assert_eq!(linear["crn"], "rotativo", "protocolo did not set crn -- default must be rotativo");
    assert_eq!(linear["campeao"], "ultimo", "protocolo did not set campeao -- default must be ultimo");
    assert!(linear["W"].get("0").is_some(), "W must carry seed 0's champion");
    assert_eq!(linear["W"]["0"].as_array().unwrap().len(), NP);

    fs::remove_dir_all(&scratch).ok();
}

#[test]
fn taxi_evolve_refuses_a_campeao_with_the_wrong_np() {
    let trace = Trace::load(&fixture_path()).expect("fixture must load");
    let scratch = scratch_dir("bad_np");

    let cenario_path = scratch.join("cenario.json");
    let cenario = serde_json::json!({
        "mundo": { "L": trace.l, "horizonte": "aberto", "T_max": 50 },
        "protocolo": { "P": 4, "G": 1, "R": 1, "seeds": [0] }
    });
    fs::write(&cenario_path, serde_json::to_string_pretty(&cenario).unwrap()).unwrap();

    let campeao_path = scratch.join("campeao_bad.json");
    let campeao = serde_json::json!({ "bracos": { "linear": { "W": { "0": vec![0.0f32; 39] } } } });
    fs::write(&campeao_path, serde_json::to_string_pretty(&campeao).unwrap()).unwrap();

    let args: Vec<String> = vec![
        "--cenario".into(),
        cenario_path.display().to_string(),
        "--mapa".into(),
        fixture_path().display().to_string(),
        "--campeao".into(),
        campeao_path.display().to_string(),
        "--saida".into(),
        scratch.join("should_not_be_written.json").display().to_string(),
    ];

    let err = taxi_world::taxi_evolve_cli::run(&args).expect_err("a 39-weight champion must be refused, not silently truncated");
    assert!(err.contains("NP=40"), "error must name the NP mismatch, got: {err}");
    assert!(
        !scratch.join("should_not_be_written.json").exists(),
        "a refused run must not write a partial output file"
    );

    fs::remove_dir_all(&scratch).ok();
}

/// Not a correctness test -- a MEASUREMENT. `#[ignore]`d so the default
/// `cargo test -p taxi-world` suite stays fast; run explicitly:
/// `CARGO_BUILD_JOBS=4 cargo test -p taxi-world --release --test evolve_smoke \
///  -- --ignored --nocapture timing_one_generation_p128_r4`.
#[test]
#[ignore = "perf measurement -- run with --release --ignored --nocapture, see this test's own doc comment"]
fn timing_one_generation_p128_r4() {
    let trace = Trace::load(&fixture_path()).expect("fixture must load");
    let world = trace.to_world();
    let cfg = Config::default();

    // 128 genomes, varied (not all-identical -- a flat population would let a
    // dead-simple branch predictor skew the number).
    let genomes: Vec<Genome> = (0..128)
        .map(|p| Genome((0..NP).map(|i| ((p * 7 + i * 3) % 17) as f32 * 0.05 - 0.4).collect()))
        .collect();

    let mut sim = SimCpu::new();
    let t0 = std::time::Instant::now();
    let fitness = sim.avaliar(&genomes, &[world], &cfg, 7200, 777_000, 4);
    let elapsed = t0.elapsed();

    assert_eq!(fitness.len(), 128);
    eprintln!(
        "P=128 R=4 T=7200 NC=1 (one GA generation): {:.3} s total, {:.2} ms/genome, {:.0} being-ticks/s",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / 128.0,
        (128.0 * 4.0 * 7200.0) / elapsed.as_secs_f64(),
    );
}
