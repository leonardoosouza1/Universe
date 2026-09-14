//! Unit tests of the reflex-readout block (`--fly-readout reflex`,
//! `docs/FLY-MAP-BRAIN.md` "Reflex readout") — the v3 trailing block, its host-side
//! override of `out_pair`, and the two pure functions `context.rs`'s FLY BRAIN OVERRIDE
//! site calls.
//!
//! A separate file, same reason as `fly_map_eye_tests.rs`: keeps `fly_map_eye.rs` under
//! the 600-line cap of `docs/STYLE.md` §2. Every test here runs on the CPU; the GPU
//! kernel needs no edit for this feature (it already reads `out_pair` generically — see
//! `gpu::fly_map::parity_test::fly_map_parity_with_reflex_readout`, `#[ignore]`d).

use std::io::Read;

use super::*;

/// Same policy as `fly_map_eye_tests.rs::scratch_dir`.
fn scratch_dir() -> std::path::PathBuf {
    let dedicated = std::path::PathBuf::from("/media/llm-data/scratch/universe-fly-map-tests");
    if std::fs::create_dir_all(&dedicated).is_ok() {
        return dedicated;
    }
    std::env::temp_dir()
}

// ── flymap v3: the reflex-readout block round-trips through disk ──────────────────

/// A map carrying the reflex-readout block writes the v3 magic (`FLYMAP02`, shared with
/// v2 — the magic only distinguishes "any trailing block" from "none") with flag bit 3
/// set, and every one of the six group indices — including the `NONE` throttle pair —
/// survives the disk exactly.
#[test]
fn reflex_block_round_trips_through_disk() {
    let map = FlyMap::synthetic_with_reflex();
    assert!(
        map.has_reflex_readout(),
        "the map under test must carry the block"
    );
    let path = scratch_dir().join("flymap_reflex_round_trip.bin");
    let path = path.to_str().unwrap().to_string();
    map.write(&path).expect("wrote the v3 map");

    let back = FlyMap::load(&path).expect("read the v3 map back");
    let _ = std::fs::remove_file(&path);

    assert!(back.has_reflex_readout(), "the block must survive the disk");
    assert_eq!(
        back.reflex_out,
        Some([NONE, NONE, 0, 1, 1, 0]),
        "all six group indices, including NONE, must round-trip exactly"
    );
    // The other three v2 blocks were never set on this map — confirms the reflex block
    // does not accidentally drag flag bits 0-2 along with it.
    assert!(back.pre_csr.is_none());
    assert!(back.pixel_of_neuron.is_none());
    assert!(back.no_refractory.is_none());
}

/// A map with no trailing block at all (`FlyMap::synthetic`) still loads, with
/// `reflex_out` `None` — the same "v1 body unchanged" guarantee the other three blocks
/// already have. `--fly-readout reflex` must refuse a map like this one.
#[test]
fn a_map_without_the_reflex_block_still_loads() {
    let map = FlyMap::synthetic();
    let path = scratch_dir().join("flymap_no_reflex_block.bin");
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
        "no trailing block must write the historical v1 magic"
    );

    let back = FlyMap::load(&path).expect("a v1 file must still load");
    let _ = std::fs::remove_file(&path);
    assert!(back.reflex_out.is_none());
    assert!(!back.has_reflex_readout());
}

/// A truncated reflex-readout block is refused with a message naming it, same contract
/// as the other three trailing blocks (`fly_map_eye_tests.rs`,
/// `a_truncated_pixel_binding_block_is_refused_and_names_it`).
#[test]
fn a_truncated_reflex_block_is_refused_and_names_it() {
    let map = FlyMap::synthetic_with_reflex();
    let path = scratch_dir().join("flymap_truncated_reflex_block.bin");
    let path = path.to_str().unwrap().to_string();
    map.write(&path)
        .expect("wrote a v3 map with the reflex block");

    // OUT_SLOTS = 6 words = 24 bytes; chop 2 to leave a torn word at the tail.
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, &full[..full.len() - 2]).unwrap();

    let err = FlyMap::load(&path).expect_err("a truncated trailing block must be refused");
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("reflex_out"),
        "the error must name the block that was truncated, got: {err}"
    );
}

/// `validate()` refuses a reflex block naming a group outside `0..k_out` that is not
/// `NONE` — the same defensive check the other group tables (`in_group_of`,
/// `out_members`) already get.
#[test]
fn validate_refuses_a_reflex_group_outside_k_out() {
    let mut map = FlyMap::synthetic(); // k_out = 2
    map.reflex_out = Some([NONE, NONE, 0, 1, 99, 0]);
    let path = scratch_dir().join("flymap_reflex_bad_group.bin");
    let path = path.to_str().unwrap().to_string();
    // `write` validates before it writes (`fly_map.rs`, `self.validate(path)?` at the top of
    // `write`), so a map that could never be loaded is never produced in the first place —
    // the refusal is asserted here rather than on the round trip.
    let err = map
        .write(&path)
        .expect_err("group 99 is outside 0..2 and must be refused");
    let _ = std::fs::remove_file(&path);
    assert!(err.to_string().contains("reflex_out"), "got: {err}");
}

// ── The host-side override (`fly_map_selection::apply_reflex_readout`) ────────────

/// The packed selection carries the block's indices for EVERY being, and the genome's
/// own `out_pair` genes — the `Vec<f32>` blocks `pack_selection` read — are never
/// touched: only the packed copy (`FlySelection::out_pair`) changes.
#[test]
fn apply_reflex_readout_overwrites_every_being_and_leaves_the_genome_alone() {
    let map = FlyMap::synthetic_with_reflex();
    let layout = map.fly_layout();
    let reflex_out = map.reflex_out.expect("map carries the block");

    // Three beings, three DIFFERENT genome out_pair choices — none of them equal to
    // `reflex_out`, so the assertion below cannot pass by coincidence.
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..3usize {
        let mut block = vec![-1.0f32; layout.block_len()];
        for s in 0..OUT_SLOTS {
            block[OFF_OUT_PAIR + s] = ((s + b) % 2) as f32;
        }
        block[OFF_GAIN] = 1.0;
        block[OFF_SCALE] = 1.0;
        for t in 0..layout.n_types {
            block[OFF_TYPE_MASK + t] = 1.0;
        }
        blocks.push(block);
    }
    let blocks_before = blocks.clone();
    let refs: Vec<&[f32]> = blocks.iter().map(|v| v.as_slice()).collect();
    let mut sel = pack_selection(&refs, layout, blocks.len());
    let genome_out_pair = sel.out_pair.clone();

    apply_reflex_readout(&mut sel, reflex_out);

    assert_ne!(
        sel.out_pair, genome_out_pair,
        "the genome blocks were built to disagree with reflex_out, so the override must \
         have changed something"
    );
    for b in 0..3usize {
        assert_eq!(
            &sel.out_pair[b * OUT_SLOTS..b * OUT_SLOTS + OUT_SLOTS],
            &reflex_out,
            "being {b}: every being must carry the SAME six reflex indices"
        );
    }
    assert_eq!(
        blocks, blocks_before,
        "apply_reflex_readout must never touch the genome blocks it was packed from"
    );
}

// ── The unit-testable gates `context.rs` and `main()` consult ─────────────────────

/// `--fly-readout reflex` requires a map, and requires that map to carry the block —
/// unit-testable without spawning a `gpu_evolve` process (`main()` prints the `Err` and
/// exits non-zero on it).
#[test]
fn readout_reflex_requires_a_map_and_the_block() {
    assert!(
        readout_reflex_requires_the_block(None).is_err(),
        "no map at all must refuse"
    );
    let without = FlyMap::synthetic();
    assert!(
        readout_reflex_requires_the_block(Some(&without)).is_err(),
        "a map without the block must refuse"
    );
    let with = FlyMap::synthetic_with_reflex();
    assert!(
        readout_reflex_requires_the_block(Some(&with)).is_ok(),
        "a map carrying the block must be accepted"
    );
}

/// [`reflex_throttle_is_none`] reads ONLY the throttle pair (slots 0, 1) — a NONE steer
/// or brake slot, which the real export never produces but the format allows, must not
/// be mistaken for a NONE throttle.
#[test]
fn reflex_throttle_is_none_reads_the_throttle_pair_only() {
    assert!(reflex_throttle_is_none([NONE, NONE, 0, 1, 1, 0]));
    assert!(!reflex_throttle_is_none([0, 1, NONE, NONE, 1, 0]));
    assert!(!reflex_throttle_is_none([0, NONE, NONE, NONE, NONE, NONE]));
}

/// [`apply_reflex_host_override`]: throttle is substituted ONLY when
/// `throttle_is_none`, steer is scaled by `sign` and NOTHING else is, brake always
/// passes through.
#[test]
fn apply_reflex_host_override_substitutes_throttle_only_when_none_and_scales_only_steer() {
    let cmd = [0.9f32, 0.4, 0.2]; // [throttle, steer, brake] as fly_read/cpu_read order it

    let substituted = apply_reflex_host_override(cmd, true, 0.6, 1.0);
    assert_eq!(
        substituted,
        [0.6, 0.4, 0.2],
        "NONE throttle must be replaced by `cruise`"
    );

    let passthrough = apply_reflex_host_override(cmd, false, 0.6, 1.0);
    assert_eq!(
        passthrough, cmd,
        "a bound throttle must pass through unchanged"
    );

    let flipped = apply_reflex_host_override(cmd, false, 0.6, -1.0);
    assert_eq!(
        flipped,
        [cmd[0], -cmd[1], cmd[2]],
        "sign = -1 must flip steer only, leaving throttle and brake untouched"
    );

    let default_arm = apply_reflex_host_override(cmd, false, 0.6, 1.0);
    assert_eq!(
        default_arm, cmd,
        "throttle_is_none = false and sign = 1.0 (the defaults) must be a no-op — the \
         reflex arm off must reproduce the historical numbers bit for bit"
    );
}
