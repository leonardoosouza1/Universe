//! Regression tests for genome serialization — the barrier against Gap #7 of the
//! project's CLAUDE.md ("new genome field without updating the disk"). Do NOT remove
//! (rule 9 of REFACTOR_PLAN.md §5).

use crate::genome_io::{load_genome, save_genome};
use universe::gpu::evolution::Genome;

/// Genome round trip: what comes off the disk has to be what went in.
///
/// # Why this test exists
///
/// `save_genome` wrote L1, L2 and KAN. When the slow brain entered the genome
/// (1091 weights) and later the τ bank (96), the serializer was not updated — and the
/// champion silently lost those fields on every `--carregar`, coming back with the
/// default initialization, identical for the whole population. Generations of selection on
/// modulation were discarded with no symptom: nothing failed, nothing warned, the
/// numbers were just worse than they should have been.
///
/// No amount of code reading catches this, because each function did exactly what it
/// promised. What catches it is comparing the two sides of the disk field by field. This test
/// fails automatically when someone adds the NEXT field to the genome and forgets
/// the serializer — which is the real mistake, and it is going to happen again.
#[test]
fn genoma_sobrevive_ida_e_volta_no_disco() {
    let mut original = Genome::random(12345);
    original.randomize_tau(999);
    // Marks each block with a value impossible to come out of the default initialization, so
    // that "passed" cannot mean "both sides fell into the same default".
    original.slow[0] = -1.234_5;
    let ultimo_slow = original.slow.len() - 1;
    original.slow[ultimo_slow] = 0.876_5;
    original.l2[7] = -0.999;
    // SENSORS PHASE: marks outside the SENSOR_DEFAULTS values (which is what
    // `Genome::random` produces with SENSORS_EVOLVE_ON off by default) — without this the
    // test would pass even if `save_genome`/`load_genome` forgot the field, because
    // both sides would fall into the same default (the exact mistake of gap 7 in CLAUDE.md).
    original.sensors[0] = 1.111_1;
    let ultimo_sensor = original.sensors.len() - 1;
    original.sensors[ultimo_sensor] = 199.9;
    // P1.7 (asynchronous brain): marks outside the neutrals of `Genome::random`
    // (0.0/1.0/0.0/0.0) — same care as the rest of the test: without this the test
    // would pass even if `save_genome`/`load_genome` forgot the fields,
    // because both sides would fall into the same default (the exact mistake of gap 7).
    original.neuron_alpha[0] = 0.123_4;
    let ultimo_alpha = original.neuron_alpha.len() - 1;
    original.neuron_alpha[ultimo_alpha] = 0.876_5;
    original.neuron_period_l1[0] = 5.0;
    let ultimo_periodo = original.neuron_period_l1.len() - 1;
    original.neuron_period_l1[ultimo_periodo] = 7.0;
    original.neuron_threshold_l1[0] = 0.321;
    let ultimo_limiar = original.neuron_threshold_l1.len() - 1;
    original.neuron_threshold_l1[ultimo_limiar] = 0.654;
    original.gates_w[0] = 0.777; // M4: marks outside the neutral (0.0)
    original.gates_w[15] = -0.555;
    original.neuron_activation[0] = 2.0;
    let ultima_ativacao = original.neuron_activation.len() - 1;
    original.neuron_activation[ultima_ativacao] = 3.0;
    // C1 (sector hippocampus): marks outside the neutral (all zeros). Both ENDS of the
    // block, because a serializer that writes the right COUNT of floats from the wrong
    // offset passes a test that only checks the first one.
    original.hip[0] = 0.246_8;
    let ultimo_hip = original.hip.len() - 1;
    original.hip[ultimo_hip] = -1.357_9;

    let path = std::env::temp_dir().join("universe_genoma_ida_volta.bin");
    let path = path.to_str().unwrap();
    save_genome(&original, path).expect("wrote");
    let lido = load_genome(path).expect("read");
    let _ = std::fs::remove_file(path);

    assert_eq!(lido.l1, original.l1, "L1 changed on disk");
    assert_eq!(lido.l2, original.l2, "L2 changed on disk");
    assert_eq!(lido.kan_knots, original.kan_knots, "KAN changed on disk");
    assert_eq!(
        lido.slow, original.slow,
        "SLOW brain did not survive the disk — this was exactly the bug"
    );
    assert_eq!(lido.tau, original.tau, "τ bank did not survive the disk");
    assert_eq!(
        lido.sensors, original.sensors,
        "sensor bank did not survive the disk"
    );
    assert_eq!(
        lido.neuron_alpha, original.neuron_alpha,
        "per-neuron memory (P1.7, mechanism 1) did not survive the disk"
    );
    assert_eq!(
        lido.neuron_period_l1, original.neuron_period_l1,
        "per-neuron period (P1.7, mechanism 2) did not survive the disk"
    );
    assert_eq!(
        lido.neuron_threshold_l1, original.neuron_threshold_l1,
        "per-neuron threshold (P1.7, mechanism 2) did not survive the disk"
    );
    assert_eq!(
        lido.neuron_activation, original.neuron_activation,
        "per-neuron activation (P1.7, mechanism 3) did not survive the disk"
    );
    assert_eq!(
        lido.gates_w, original.gates_w,
        "M4: conductor gates did not survive the disk"
    );
    assert_eq!(
        lido.hip, original.hip,
        "C1: sector hippocampus did not survive the disk"
    );
    assert_eq!(
        lido.hip.len(),
        universe::gpu::hippocampus::N_HIP_GENES,
        "C1: the hippocampus block has to be 29 genes — 27 for W plus rate plus gain"
    );
}

/// A genome written by an old version (with neither slow brain nor τ) still loads.
///
/// Compatibility here is by LENGTH, not by header: `load_genome` only reads
/// the trailing blocks if the file is big enough. Without this test, whoever touches the
/// format breaks every saved champion and only finds out on the next resume.
#[test]
fn genoma_antigo_sem_lento_nem_tau_ainda_carrega() {
    use std::io::Write;
    let original = Genome::random(777);
    let path = std::env::temp_dir().join("universe_genoma_antigo.bin");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        for w in &original.l1 {
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        for w in &original.l2 {
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        for row in &original.kan_knots {
            for k in row {
                f.write_all(&k.to_le_bytes()).unwrap();
            }
        }
    }
    let lido = load_genome(path.to_str().unwrap()).expect("old file must load");
    let _ = std::fs::remove_file(&path);

    assert_eq!(lido.l1, original.l1);
    // Without the block in the file, falling into the default is the truth — that genome
    // really did not carry the data. What it must not do is read garbage or return an empty vector.
    assert_eq!(lido.tau.len(), universe::gpu::evolution::N_CFC_CHANNELS);
    assert!(lido.tau.iter().all(|&d| d == universe::cfc::DECAY_DEFAULT));
    assert_eq!(
        lido.slow.len(),
        universe::engine::SlowBrain::new().all_weights().len()
    );
    // SENSORS PHASE: a legacy genome (written before this field existed) has to land
    // exactly on SENSOR_DEFAULTS — "today's values as the default", an explicit
    // requirement of the task (item 4).
    assert_eq!(
        lido.sensors,
        universe::gpu::evolution::SENSOR_DEFAULTS.to_vec(),
        "legacy genome without sensors should load with SENSOR_DEFAULTS"
    );
    // P1.7: same requirement — a genome written before the asynchronous brain existed
    // loads with the NEUTRALS of `Genome::random` (0.0/1.0/0.0/0.0), reproducing
    // the real behaviour of that genome (it ran without the three mechanisms).
    let n_l1 = universe::gpu::evolution::N_L1_NEURONS;
    let n_l2 = universe::gpu::evolution::N_L2_NEURONS;
    assert_eq!(
        lido.neuron_alpha,
        vec![0.0; n_l1 + n_l2],
        "legacy genome without P1.7 should load alpha=0.0"
    );
    assert_eq!(
        lido.neuron_period_l1,
        vec![1.0; n_l1],
        "legacy genome without P1.7 should load period=1.0"
    );
    assert_eq!(
        lido.neuron_threshold_l1,
        vec![0.0; n_l1],
        "legacy genome without P1.7 should load threshold=0.0"
    );
    assert_eq!(
        lido.neuron_activation,
        vec![0.0; n_l1 + n_l2],
        "legacy genome without P1.7 should load activation=0.0 (tanh)"
    );
    // C1: same requirement one block further down. All-zero is not merely a safe default
    // here, it is the exact behaviour that genome ran with — zero genes give a bias of
    // [0, 0, 0], which IS "no hippocampus".
    assert_eq!(
        lido.hip,
        vec![0.0; universe::gpu::hippocampus::N_HIP_GENES],
        "legacy genome without C1 should load hip = zeros (bias [0,0,0])"
    );
}

/// A genome saved by the CURRENT format minus only the last block still loads, with the
/// hippocampus neutral.
///
/// `genoma_antigo_sem_lento_nem_tau_ainda_carrega` above truncates the file right after  // check-language: allow -- quoting an existing test name verbatim
/// KAN, which is far enough back that every optional block falls back together. This one
/// truncates by exactly the last block, which is the case that actually ships: every
/// champion `.bin` written before C1 existed is this file. If `load_genome` read the
/// hippocampus from the wrong offset it would fabricate 29 genes out of whatever came
/// before, and a resumed champion would start biasing its commands with garbage.
#[test]
fn genome_without_the_hippocampus_block_loads_it_neutral() {
    let mut original = Genome::random(4242);
    original.gates_w[0] = 0.777;
    original.gates_w[15] = -0.555;
    // What a pre-C1 `save_genome` produced: the same bytes, minus this block.
    original.hip = Vec::new();

    let path = std::env::temp_dir().join("universe_genome_without_hip.bin");
    let path = path.to_str().unwrap();
    save_genome(&original, path).expect("wrote");
    let lido = load_genome(path).expect("read");
    let _ = std::fs::remove_file(path);

    assert_eq!(
        lido.hip,
        vec![0.0; universe::gpu::hippocampus::N_HIP_GENES],
        "pre-C1 genome must load with a neutral hippocampus, never with garbage"
    );
    // The block BEFORE it has to be intact — that is what proves the fallback did not
    // consume bytes that belonged to someone else.
    assert_eq!(
        lido.gates_w, original.gates_w,
        "M4 gates corrupted by the hippocampus fallback"
    );
    // And the bias that neutral genome produces is exactly zero, for any memory.
    let mem = universe::gpu::hippocampus::SectorMemory::new();
    let b = universe::gpu::hippocampus::bias(&lido.hip, &mem.read(0));
    assert_eq!(
        b,
        [0.0, 0.0, 0.0],
        "neutral hippocampus must not bias anything"
    );
}

/// The library's `N_SECTORS` and this binary's `N_CHECKPOINTS` are the same grid written
/// in two places (the library cannot import a binary's constant). `docs/STYLE.md` §4 is
/// the rule; `WAVE_U32S`, declared six times and missed twice, is why the rule exists.
#[test]
fn sector_count_matches_the_checkpoint_grid() {
    assert_eq!(
        universe::gpu::hippocampus::N_SECTORS as u32,
        crate::statics::N_CHECKPOINTS,
        "the hippocampus indexes sectors; if the checkpoint grid changes, it has to follow"
    );
}

/// `sector_of` is the single definition of a formula that used to be written out twice —
/// once in `avaliar_held_out` and once in the training loop of `main()`. This pins its two
/// edges: progress 0 is sector 0,
/// and progress one tick short of a full lap is the last sector, never sector 24.
#[test]
fn sector_of_covers_the_lap_without_overflowing_the_grid() {
    let lap = 2928.0_f32;
    assert_eq!(crate::statics::sector_of(0.0, lap), 0);
    assert_eq!(crate::statics::sector_of(lap / 24.0, lap), 1);
    assert_eq!(
        crate::statics::sector_of(lap * 0.999, lap),
        crate::statics::N_CHECKPOINTS - 1
    );
}

/// A v2 genome (with the `UGEN` header) survives the disk round trip.
///
/// Complements `genoma_sobrevive_ida_e_volta_no_disco`: that one tests the format
/// WITHOUT a header (legacy). This one guarantees that the new versioned format (magic +
/// version) is also read back correctly by the same `load_genome`.
#[test]
fn genoma_v2_com_cabecalho_sobrevive_ida_e_volta() {
    let mut original = Genome::random(2026);
    original.randomize_tau(42);
    original.l1[0] = 0.123_4;
    original.slow[3] = -0.567_8;
    original.sensors[1] = 88.8; // SENSORS PHASE: marks outside SENSOR_DEFAULTS
    original.neuron_alpha[2] = 0.444_4; // P1.7: marks outside the neutrals
    original.neuron_period_l1[2] = 6.0;
    original.neuron_threshold_l1[2] = 0.222_2;
    original.neuron_activation[2] = 1.0;

    let path = std::env::temp_dir().join("universe_genoma_v2_cabecalho.bin");
    let path_str = path.to_str().unwrap();
    save_genome(&original, path_str).expect("wrote");

    // Confirms the file has the magic "UGEN" + version 2 at the start — without this the
    // test could pass even if `save_genome` had gone back to writing without a
    // header (load_genome tolerates both formats by sniffing).
    let bytes = std::fs::read(path_str).expect("read raw bytes");
    assert_eq!(
        &bytes[0..4],
        b"UGEN",
        "magic header missing in the saved file"
    );
    let versao = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(versao, 2, "header version is not 2");

    let lido = load_genome(path_str).expect("read");
    let _ = std::fs::remove_file(path_str);

    assert_eq!(lido.l1, original.l1, "L1 changed on disk (v2)");
    assert_eq!(lido.l2, original.l2, "L2 changed on disk (v2)");
    assert_eq!(
        lido.kan_knots, original.kan_knots,
        "KAN changed on disk (v2)"
    );
    assert_eq!(lido.slow, original.slow, "slow brain changed on disk (v2)");
    assert_eq!(lido.tau, original.tau, "τ changed on disk (v2)");
    assert_eq!(
        lido.sensors, original.sensors,
        "sensors changed on disk (v2)"
    );
    assert_eq!(
        lido.neuron_alpha, original.neuron_alpha,
        "memory (P1.7) changed on disk (v2)"
    );
    assert_eq!(
        lido.neuron_period_l1, original.neuron_period_l1,
        "period (P1.7) changed on disk (v2)"
    );
    assert_eq!(
        lido.neuron_threshold_l1, original.neuron_threshold_l1,
        "threshold (P1.7) changed on disk (v2)"
    );
    assert_eq!(
        lido.neuron_activation, original.neuron_activation,
        "activation (P1.7) changed on disk (v2)"
    );
}

/// A SYNTHETIC legacy blob (no "UGEN" magic, only the raw bytes of L1|L2|KAN) still
/// loads through the same `load_genome` that reads the v2 format.
///
/// Unlike `genoma_antigo_sem_lento_nem_tau_ainda_carrega` (check-language: allow --
/// quotes a test identifier still waiting on the rename wave) which also tests
/// compatibility, this test is E3's explicit request: "the loader accepts v2 AND
/// legacy without magic (sniff the first 4 bytes)" — built from pure helpers, without
/// depending on any real file written by an old version of the binary.
#[test]
fn load_genome_detecta_legado_por_sniff_dos_primeiros_4_bytes() {
    let original = Genome::random(31337);
    let path = std::env::temp_dir().join("universe_genoma_legado_sniff.bin");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        for w in &original.l1 {
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        for w in &original.l2 {
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        for row in &original.kan_knots {
            for k in row {
                f.write_all(&k.to_le_bytes()).unwrap();
            }
        }
    }
    // Confirms the first 4 bytes are NOT the "UGEN" magic — the blob is
    // genuinely legacy, starting with arbitrary f32 weight bytes.
    let bytes = std::fs::read(&path).unwrap();
    assert_ne!(
        &bytes[0..4],
        b"UGEN",
        "the test blob should be legacy (no magic)"
    );

    let lido = load_genome(path.to_str().unwrap()).expect("synthetic legacy blob must load");
    let _ = std::fs::remove_file(&path);

    assert_eq!(lido.l1, original.l1);
    assert_eq!(lido.l2, original.l2);
    assert_eq!(lido.kan_knots, original.kan_knots);
}

/// P0.5 (spectral initialization of L1): a LEGACY genome loaded from disk is NOT
/// re-scaled — inheritance untouched.
///
/// `Genome::random` (the source of the spectral normalization, `src/gpu/evolution.rs`) only runs
/// on the CREATION of a new genome. `load_genome` never calls `Genome::random` — it builds the
/// `Genome` directly by struct literal from the bytes read (genome_io.rs). This
/// test simulates exactly the real case: a `.bin` saved BEFORE the P0.5 hypothesis existed,
/// with L1 at the raw spectral gain that σ=0.30 i.i.d. always produced (~7.0, see
/// CLAUDE.md/docs/PLANO_EVOLUCAO.md §P0.5) — far from the target of 1.0. If `load_genome`
/// ever started "correcting" the genome on load, this test catches it: the champion
/// saved by generations of selection would change behaviour in the middle of the battery, with no
/// warning, when resumed.
#[test]
fn genoma_legado_carregado_do_disco_nao_e_reescalado() {
    use universe::config::iara::brain::{L1_NEURONS, SENSORY_WAVE_SIZE};
    use universe::math::linalg::largest_singular_value;

    // Simulates the "legacy" L1: a large, known spectral gain, produced without
    // any normalization (that is how EVERY genome was born before P0.5).
    let mut legado = Genome::random(9999);
    for w in legado.l1.iter_mut() {
        *w *= 6.0;
    }
    let ganho_antes = largest_singular_value(&legado.l1, L1_NEURONS, SENSORY_WAVE_SIZE);
    assert!(
        ganho_antes > 2.0,
        "test precondition: gain should be far from 1.0, got {ganho_antes}"
    );

    let path = std::env::temp_dir().join("universe_genoma_legado_ganho_p05.bin");
    let path_str = path.to_str().unwrap();
    save_genome(&legado, path_str).expect("wrote");
    let lido = load_genome(path_str).expect("read");
    let _ = std::fs::remove_file(path_str);

    assert_eq!(
        lido.l1, legado.l1,
        "L1 changed on the disk round trip — it should be byte-identical"
    );
    let ganho_depois = largest_singular_value(&lido.l1, L1_NEURONS, SENSORY_WAVE_SIZE);
    assert!(
        (ganho_depois - ganho_antes).abs() < 1e-3,
        "spectral gain changed on load: before={ganho_antes}, after={ganho_depois} — \
         the legacy genome was re-scaled, and it should not have been"
    );
}
