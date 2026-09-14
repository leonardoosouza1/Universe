//! Unit tests of the flymap v2 trailing blocks and the eye binding (`fly_map_eye.rs`).
//!
//! A separate file (`#[path]`, the pattern `fly_map.rs` already uses for
//! `fly_map_tests.rs`) so `fly_map_eye.rs` stays under the 600-line cap of
//! `docs/STYLE.md` §2, along the same map/eye seam the production files split on.
//!
//! Every test here runs on the CPU; the GPU kernel's eye extension is checked against
//! [`eye_drive`] by `gpu::fly_map::parity_test`, which is `#[ignore]`d.

use std::io::Read;

use super::*;

/// Where a test may write its synthetic map — same policy as `fly_map_tests.rs`'s
/// `scratch_dir` (`/media/llm-data/scratch` first, the storage policy in the project
/// `CLAUDE.md`; falls back to the system temp dir so the suite still runs without that
/// partition mounted).
fn scratch_dir() -> std::path::PathBuf {
    let dedicated = std::path::PathBuf::from("/media/llm-data/scratch/universe-fly-map-tests");
    if std::fs::create_dir_all(&dedicated).is_ok() {
        return dedicated;
    }
    std::env::temp_dir()
}

/// A selection for `n_beings` from one block per being, against the synthetic layout.
/// Duplicated from `fly_map_tests.rs` rather than shared: each test file is meant to be
/// readable on its own, the same reason `gpu/fly_map/parity_test.rs` keeps its own
/// `synthetic_wave`/`pack_wave` instead of importing them.
fn selection_from(blocks: &[Vec<f32>], layout: FlyLayout) -> FlySelection {
    let refs: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
    pack_selection(&refs, layout, blocks.len())
}

/// A block with every gene set by hand: nothing in these tests may depend on an RNG.
fn block_of(layout: FlyLayout, binds: &[(usize, i32)], out_pair: [i32; OUT_SLOTS]) -> Vec<f32> {
    let mut b = vec![-1.0f32; layout.block_len()];
    for ch in 0..IN_CHANNELS {
        b[ch] = -1.0;
    }
    for &(ch, g) in binds {
        b[ch] = g as f32;
    }
    for (s, g) in out_pair.iter().enumerate() {
        b[OFF_OUT_PAIR + s] = *g as f32;
    }
    b[OFF_GAIN] = 1.0;
    b[OFF_SCALE] = 1.0;
    for t in 0..layout.n_types {
        b[OFF_TYPE_MASK + t] = 1.0;
    }
    b
}

// ── flymap v2: trailing blocks and the eye binding (docs/FLY-MAP-BRAIN.md §1, "Eye
// binding") ──────────────────────────────────────────────────────────────────────

/// A map with no trailing block writes the historical v1 file byte for byte, and that
/// file loads through the SAME reader `FlyMap::load` now uses for v2 — the whole point
/// of a magic-gated optional tail is that an old export never has to be re-exported.
#[test]
fn a_v1_file_still_loads_through_the_v2_reader_with_every_v2_field_absent() {
    let map = FlyMap::synthetic();
    let path = scratch_dir().join("flymap_v1_explicit.bin");
    let path = path.to_str().unwrap().to_string();
    map.write(&path)
        .expect("wrote a v1 map (no trailing blocks)");

    let mut magic = [0u8; 8];
    std::fs::File::open(&path)
        .unwrap()
        .read_exact(&mut magic)
        .unwrap();
    assert_eq!(
        &magic, MAGIC_V1,
        "a map with no trailing block must write the historical FLYMAP01 magic, byte for \
         byte — that is what makes an old export loadable without re-running the exporter"
    );

    let back = FlyMap::load(&path).expect("a v1 file must load through the same loader as v2");
    let _ = std::fs::remove_file(&path);
    assert!(
        back.pre_csr.is_none(),
        "a v1 file carries no presynaptic CSR"
    );
    assert!(
        back.pixel_of_neuron.is_none(),
        "a v1 file carries no pixel binding"
    );
    assert!(
        back.no_refractory.is_none(),
        "a v1 file carries no no-refractory bitmap"
    );
    assert!(
        !back.has_pixel_binding(),
        "--fly-eye must refuse a map like this one (no pixel-binding block)"
    );
}

/// A v1 magic with a nonzero `flags` word is a contradiction (v1 never sets a v2 bit) —
/// caught at load, not silently misparsed as "no trailing blocks, but with garbage flags
/// nobody checked".
#[test]
fn a_v1_magic_with_a_nonzero_flags_word_is_rejected() {
    let map = FlyMap::synthetic();
    let path = scratch_dir().join("flymap_v1_bad_flags.bin");
    let path = path.to_str().unwrap().to_string();
    map.write(&path).expect("wrote a v1 map");

    // Header word 6 (flags) sits right after the 8-byte magic and the first 6 header
    // words (n, m, n_types, k_in, k_out, out_members_len): offset 8 + 6*4 = 32.
    let mut bytes = std::fs::read(&path).unwrap();
    let off = 8 + 6 * 4;
    bytes[off..off + 4].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let err = FlyMap::load(&path).expect_err("a v1 magic with a set flag bit must be refused");
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("flags"),
        "the error must name the contradiction, got: {err}"
    );
}

/// The v2 round trip: a map carrying all three trailing blocks writes `FLYMAP02` and every
/// block survives the disk exactly, in the fixed order of §1.
#[test]
fn flymap_v2_round_trip_carries_all_three_trailing_blocks() {
    let mut map = FlyMap::synthetic();
    // `FlyMap::synthetic` has m = 9 edges (rows of 1, 1, 2, 1, 2, 2); the presynaptic CSR
    // re-indexes the SAME 9 edges by their presynaptic neuron, so `indices`/`weights` are
    // `m` long and `indptr` (7 = n + 1 entries) must end at 9 — `validate` checks both.
    let pre_csr = PresynapticCsr {
        indptr: vec![0, 1, 2, 4, 5, 7, 9],
        indices: vec![5, 0, 1, 2, 3, 1, 4, 0, 1],
        weights: vec![0.5, -1.0, 0.75, 0.25, -0.5, 1.0, 0.3, -0.2, 0.6],
    };
    map.pre_csr = Some(pre_csr.clone());
    let mut pixel_of_neuron = vec![NONE; map.n()];
    pixel_of_neuron[0] = 5;
    pixel_of_neuron[4] = 1763;
    map.pixel_of_neuron = Some(pixel_of_neuron.clone());
    let no_refractory = vec![0b101010u32]; // ceil(6 / 32) = 1 word
    map.no_refractory = Some(no_refractory.clone());

    let path = scratch_dir().join("flymap_v2_round_trip.bin");
    let path = path.to_str().unwrap().to_string();
    map.write(&path).expect("wrote the v2 map");

    let mut magic = [0u8; 8];
    std::fs::File::open(&path)
        .unwrap()
        .read_exact(&mut magic)
        .unwrap();
    assert_eq!(
        &magic, MAGIC_V2,
        "a map with any trailing block must write the v2 magic"
    );

    let back = FlyMap::load(&path).expect("read the v2 map back");
    let _ = std::fs::remove_file(&path);

    assert!(
        back.has_pixel_binding(),
        "--fly-eye must accept a map like this one"
    );
    let back_csr = back.pre_csr.expect("presynaptic CSR must survive the disk");
    assert_eq!(back_csr.indptr, pre_csr.indptr);
    assert_eq!(back_csr.indices, pre_csr.indices);
    assert_eq!(back_csr.weights, pre_csr.weights);
    assert_eq!(back.pixel_of_neuron, Some(pixel_of_neuron));
    assert_eq!(back.no_refractory, Some(no_refractory));
}

/// A file that declares the pixel-binding flag but is cut off mid-block is refused with a
/// message that names the block — the bare `io::Error` underneath ("failed to fill whole
/// buffer") would not tell a reader which of the three new blocks was short.
#[test]
fn a_truncated_pixel_binding_block_is_refused_and_names_it() {
    let mut map = FlyMap::synthetic();
    map.pixel_of_neuron = Some(vec![NONE; map.n()]);
    let path = scratch_dir().join("flymap_truncated_pixel_block.bin");
    let path = path.to_str().unwrap().to_string();
    map.write(&path)
        .expect("wrote a v2 map with only the pixel-binding block");

    // pixel_of_neuron is the only trailing block here, so any truncation of the tail
    // lands inside it. n = 6 neurons x 4 bytes = 24 bytes; chop 2 to leave a torn word.
    let full = std::fs::read(&path).unwrap();
    assert!(full.len() > 2, "the file must be long enough to truncate");
    std::fs::write(&path, &full[..full.len() - 2]).unwrap();

    let err = FlyMap::load(&path).expect_err("a truncated trailing block must be refused");
    let _ = std::fs::remove_file(&path);
    let msg = err.to_string();
    assert!(
        msg.contains("pixel_of_neuron"),
        "the error must name the block that was truncated, got: {msg}"
    );
}

/// [`eye_drive`] directly: dark (Road, zero range) drives at the full gain, bright (Wall,
/// zero range) drives at zero, and a neuron with no pixel binding never drives at all —
/// the three cases `docs/FLY-MAP-BRAIN.md` "Eye binding" states as the formula's shape.
#[test]
fn eye_drive_is_strong_in_the_dark_and_silent_in_the_light() {
    let pixel_of_neuron = [0u32, NONE];
    // being 0's pixel 0: Road (material 1) at 0 mm -> legacy_encoding = -128 -> brightness
    // clamps to 0 -> u_eye = gain * (1 - 0) = gain.
    // being 1's pixel 0: Wall (material 2) at 0 mm -> legacy_encoding = 127 -> brightness =
    // 1 -> u_eye = gain * (1 - 1) = 0.
    let retina = [1u32, 2u32];
    let frame = EyeFrame {
        pixel_of_neuron: &pixel_of_neuron,
        retina: &retina,
        retina_stride: 1,
    };

    let dark = eye_drive(2.0, &frame, 0, 0).expect("neuron 0 is bound");
    assert!(
        (dark - 2.0).abs() < 1e-6,
        "dark (Road at zero range) must drive at full gain, got {dark}"
    );
    let bright = eye_drive(2.0, &frame, 1, 0).expect("neuron 0 is bound");
    assert!(
        bright.abs() < 1e-6,
        "bright (Wall at zero range) must drive at zero, got {bright}"
    );
    assert!(
        eye_drive(2.0, &frame, 0, 1).is_none(),
        "neuron 1's pixel is NONE — it must not drive at all"
    );
}

/// The neuron-level replacement rule of "Eye binding": a neuron WITH a pixel binding
/// ignores the genome's channel binding entirely (even though it shares the same afferent
/// group as a neuron that has no pixel binding), and a neuron WITHOUT one is completely
/// unaffected by the eye frame — proven on tick 1, where `r_old` is still all zero, so
/// each neuron's state depends only on its own `u` term and not on any neighbour's.
#[test]
fn eye_driven_neuron_ignores_the_genome_binding_others_do_not() {
    let map = FlyMap::synthetic_with_eye(); // neuron 0 -> pixel 5, neurons 1,2,3,5 unbound... (4 -> pixel 1763)
    let layout = map.fly_layout();
    // Neurons 0 and 1 share afferent group 0 (`FlyMap::synthetic`'s `in_group_of`); bind
    // channel 0 to group 0 so BOTH would read the same genome input if neither were
    // eye-driven — the shared group is what makes divergence between them meaningful.
    let block = block_of(layout, &[(0, 0)], [0, 1, 0, 1, 0, 1]);
    let sel = selection_from(&[block], layout);
    let mut wave = vec![0i8; IN_CHANNELS];
    wave[0] = -128;

    let retina_stride = crate::config::iara::vision::RETINA_TOTAL;
    let mut retina = vec![0u32; retina_stride];
    retina[5] = 1u32; // Road at 0 mm under neuron 0's pixel binding
    let frame = EyeFrame {
        pixel_of_neuron: map
            .pixel_of_neuron
            .as_deref()
            .expect("map carries the block"),
        retina: &retina,
        retina_stride,
    };
    let alpha = 0.25f32;

    let mut x_eye = vec![0.0f32; map.n()];
    let r_old = vec![0.0f32; map.n()];
    let mut r_eye = vec![0.0f32; map.n()];
    cpu_step(
        &map,
        &sel,
        &wave,
        alpha,
        &mut x_eye,
        &r_old,
        &mut r_eye,
        Some(&frame),
    );

    let mut x_plain = vec![0.0f32; map.n()];
    let mut r_plain = vec![0.0f32; map.n()];
    cpu_step(
        &map,
        &sel,
        &wave,
        alpha,
        &mut x_plain,
        &r_old,
        &mut r_plain,
        None,
    );

    let expected_plain_u = 1.0f32 * (-128.0) / 127.0; // gain 1 * wave[0] / 127
    let expected_eye_u = 1.0f32 * (1.0 - 0.0); // gain 1 * (1 - brightness), brightness 0
    assert!(
        (x_plain[0] - alpha * expected_plain_u).abs() < 1e-6,
        "the plain run must match input_drive by hand: got {}",
        x_plain[0]
    );
    assert!(
        (x_eye[0] - alpha * expected_eye_u).abs() < 1e-6,
        "the eye run must match u_eye = gain * (1 - brightness) by hand: got {}",
        x_eye[0]
    );
    assert!(
        (x_eye[0] - x_plain[0]).abs() > 1e-3,
        "neuron 0 is eye-driven: the two runs must diverge, eye {} vs plain {}",
        x_eye[0],
        x_plain[0]
    );
    assert_eq!(
        x_eye[1], x_plain[1],
        "neuron 1 shares group 0 with neuron 0 but has NO pixel binding of its own — \
         supplying eye data for neuron 0 must not move it"
    );
    // r_old is all zero on tick 1, so this equality is not a coincidence of cancellation;
    // it is the direct consequence of neuron 1 never touching `frame` inside `eye_drive`.
    let _ = &r_eye;
    let _ = &r_plain;
}

/// The real exporter's output, not a six-neuron synthetic map — the byte-precise proof
/// that this reader matches what `neuromod-lab/src/fly_export.py` actually writes, not
/// only what this test file constructs by hand. `#[ignore]`d like `fly_map_bench`: it
/// needs the 233 MB export on the data partition, not the repository, and no GPU.
///
/// The header this asserts (n=138639 m=15091983 n_types=8866 k_in=268 k_out=516
/// out_members=1470 flags=7) is the one `docs/KERNELS-FLY-SPIKING.md` §3 records as
/// "READY, verified 14/09 12:55" for this exact file — checked here independently,
/// through `FlyMap::load` and `FlyMap::validate` rather than a hex dump.
#[test]
#[ignore = "needs the real export on /media/llm-data — run manually: \
            cargo test --release --lib -- --ignored flymap_v2_real_export"]
fn flymap_v2_real_export_loads_and_matches_the_documented_header() {
    let path = "/media/llm-data/datasets/flywire/flymap_783_v2.bin";
    if !std::path::Path::new(path).exists() {
        eprintln!("[flymap_v2_real_export] {path} not found — test skipped");
        return;
    }
    let map = FlyMap::load(path).expect("the real v2 export must load through this reader");
    assert_eq!(map.n(), 138_639);
    assert_eq!(map.m(), 15_091_983);
    assert_eq!(map.n_types, 8_866);
    assert_eq!(map.k_in, 268);
    assert_eq!(map.k_out(), 516);
    assert_eq!(map.out_members.len(), 1_470);
    // flags = 7 = every one of the three v2 bits (§1): presynaptic CSR, pixel binding,
    // no-refractory bitmap.
    assert!(
        map.pre_csr.is_some(),
        "flags bit 0 was set — the presynaptic CSR must be Some"
    );
    assert!(
        map.has_pixel_binding(),
        "flags bit 1 was set — --fly-eye must accept this file"
    );
    assert!(
        map.no_refractory.is_some(),
        "flags bit 2 was set — the no-refractory bitmap must be Some"
    );
    let pre_csr = map.pre_csr.as_ref().unwrap();
    assert_eq!(pre_csr.indptr.len(), map.n() + 1);
    assert_eq!(pre_csr.indices.len(), map.m());
    assert_eq!(pre_csr.weights.len(), map.m());
    let pixels = map.pixel_of_neuron.as_ref().unwrap();
    assert_eq!(pixels.len(), map.n());
    let bound = pixels.iter().filter(|&&p| p != NONE).count();
    assert!(
        bound > 0,
        "the exporter's eye-binding pass must have bound at least one neuron"
    );
    assert_eq!(
        map.no_refractory.as_ref().unwrap().len(),
        map.n().div_ceil(32)
    );
}

/// The gate `FLY_EYE_ON` (`--fly-eye`): with it off, an orchestration layer that has real
/// eye data available (a map with the block, a populated retina frame) still passes `None`
/// into `cpu_step`, and the result is byte-identical to calling `cpu_step` with `None`
/// directly — the exact code path this function had before the eye extension existed,
/// since the branch inside is skipped, not replaced (see `cpu_step`'s doc comment). With
/// the SAME gate on, the SAME data must move the eye-driven neuron, or the gate would be
/// provably inert.
///
/// Guard style copied from `evolve_fly_block_has_its_own_rates_and_gate`
/// (`src/gpu/evolution.rs`), including the `Drop` that restores the atomic so a failing
/// assert here does not leak `true` into tests running next to this one in the same
/// process.
#[test]
fn eye_gate_off_leaves_cpu_step_byte_identical_to_before_the_extension() {
    use std::sync::atomic::Ordering;

    struct RestoreEye(bool);
    impl Drop for RestoreEye {
        fn drop(&mut self) {
            FLY_EYE_ON.store(self.0, Ordering::SeqCst);
        }
    }
    let _guard = RestoreEye(FLY_EYE_ON.swap(true, Ordering::SeqCst));
    assert!(eye_on(), "the gate must read back what was just stored");
    FLY_EYE_ON.store(false, Ordering::SeqCst);
    assert!(!eye_on());

    let map = FlyMap::synthetic_with_eye();
    let layout = map.fly_layout();
    let block = block_of(layout, &[(0, 0)], [0, 1, 0, 1, 0, 1]);
    let sel = selection_from(&[block], layout);
    let mut wave = vec![0i8; IN_CHANNELS];
    wave[0] = 90;

    let retina_stride = crate::config::iara::vision::RETINA_TOTAL;
    let mut retina = vec![0u32; retina_stride];
    retina[5] = 1u32;
    let frame = EyeFrame {
        pixel_of_neuron: map.pixel_of_neuron.as_deref().unwrap(),
        retina: &retina,
        retina_stride,
    };

    // The call site this simulates: `context.rs::tick()` reads the gate ONCE and decides
    // what to pass; `cpu_step` itself never reads `FLY_EYE_ON`.
    let eye_arg = eye_on().then_some(&frame);
    assert!(eye_arg.is_none(), "gate is off, the caller must pass None");

    let mut x_off = vec![0.0f32; map.n()];
    let r_old = vec![0.0f32; map.n()];
    let mut r_off = vec![0.0f32; map.n()];
    cpu_step(
        &map, &sel, &wave, 0.25, &mut x_off, &r_old, &mut r_off, eye_arg,
    );

    let mut x_ref = vec![0.0f32; map.n()];
    let mut r_ref = vec![0.0f32; map.n()];
    cpu_step(
        &map, &sel, &wave, 0.25, &mut x_ref, &r_old, &mut r_ref, None,
    );

    assert_eq!(
        x_off, x_ref,
        "gate off must be byte-identical to the pre-eye formula"
    );
    assert_eq!(
        r_off, r_ref,
        "gate off must be byte-identical to the pre-eye formula"
    );

    FLY_EYE_ON.store(true, Ordering::SeqCst);
    let eye_arg_on = eye_on().then_some(&frame);
    let mut x_on = vec![0.0f32; map.n()];
    let mut r_on = vec![0.0f32; map.n()];
    cpu_step(
        &map, &sel, &wave, 0.25, &mut x_on, &r_old, &mut r_on, eye_arg_on,
    );
    assert_ne!(
        x_on[0], x_off[0],
        "gate on with the same data must move the eye-driven neuron, or the gate proves \
         nothing"
    );
}
