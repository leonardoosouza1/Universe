//! Unit tests of the fly map: file format, the rate model of §2, and the genome block.
//!
//! A separate file (`#[path]`, the pattern of `retina_digest_parity_test.rs`) so
//! `fly_map.rs` stays under the 600-line cap of `docs/STYLE.md` §2.
//!
//! Every test here runs on the CPU. The GPU kernels are checked against these same
//! functions by `gpu::fly_map::parity_test`, which is `#[ignore]`d.

use super::*;

/// Where a test may write its synthetic map.
///
/// `/media/llm-data/scratch` first, because the owner's system disk once fell to 340 MB
/// free and killed batteries in progress (project `CLAUDE.md`, storage policy). The
/// fallback keeps the suite runnable on a machine without that partition — the file is
/// under 1 KB, which the same policy allows in the system temp directory.
fn scratch_dir() -> std::path::PathBuf {
    let dedicated = std::path::PathBuf::from("/media/llm-data/scratch/universe-fly-map-tests");
    if directory_is_writable(&dedicated) {
        return dedicated;
    }
    std::env::temp_dir()
}

fn directory_is_writable(path: &std::path::Path) -> bool {
    if std::fs::create_dir_all(path).is_err() {
        return false;
    }
    let probe = path.join(format!(".write-probe-{}", std::process::id()));
    let writable = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&probe)
        .is_ok();
    let _ = std::fs::remove_file(probe);
    writable
}

/// A selection for `n_beings` from one block per being, against the synthetic layout.
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

#[test]
fn synthetic_map_survives_the_disk_round_trip() {
    let map = FlyMap::synthetic();
    let path = scratch_dir().join("flymap_synthetic_round_trip.bin");
    let path = path.to_str().expect("path is utf-8").to_string();
    map.write(&path).expect("wrote the synthetic map");
    let back = FlyMap::load(&path).expect("read the synthetic map back");
    let _ = std::fs::remove_file(&path);

    assert_eq!(back.n(), 6, "six neurons");
    assert_eq!(back.m(), map.m(), "edge count changed on disk");
    assert_eq!(back.indptr, map.indptr, "CSR offsets changed on disk");
    assert_eq!(back.indices, map.indices, "presynaptic ids changed on disk");
    assert_eq!(back.weights, map.weights, "signed weights changed on disk");
    assert_eq!(back.type_of, map.type_of, "cell types changed on disk");
    assert_eq!(
        back.in_group_of, map.in_group_of,
        "afferent groups changed on disk — NONE included"
    );
    assert_eq!(back.out_indptr, map.out_indptr);
    assert_eq!(back.out_members, map.out_members);
    assert_eq!(back.n_types, 3);
    assert_eq!(back.k_in, 2);
    assert_eq!(back.k_out(), 2);
    // The NONE sentinel is the one value a truncating writer would mangle into 0, which
    // would silently bind three neurons to afferent group 0.
    assert_eq!(back.in_group_of[5], NONE, "NONE must survive as NONE");
}

#[test]
fn a_file_without_the_magic_is_refused_and_says_what_writes_it() {
    let path = scratch_dir().join("flymap_not_a_map.bin");
    std::fs::write(&path, [0u8; 64]).expect("wrote the decoy");
    let err = FlyMap::load(path.to_str().unwrap()).expect_err("must refuse");
    let _ = std::fs::remove_file(&path);
    let msg = err.to_string();
    assert!(
        msg.contains("fly_export.py"),
        "the error must name the exporter that writes the format, got: {msg}"
    );
}

#[test]
fn a_pruned_cell_type_emits_exactly_zero() {
    let map = FlyMap::synthetic();
    let layout = map.fly_layout();
    // Bind channel 0 to afferent group 0 so neurons 0 and 1 are actually driven.
    let mut block = block_of(layout, &[(0, 0)], [0, 1, 0, 1, 0, 1]);
    // Prune type 1 (neurons 2 and 3).
    block[OFF_TYPE_MASK + 1] = 0.0;
    let sel = selection_from(&[block], layout);

    let mut wave = vec![0i8; IN_CHANNELS];
    wave[0] = 127;
    let mut x = vec![0.0f32; map.n()];
    let mut r_a = vec![0.0f32; map.n()];
    let mut r_b = vec![0.0f32; map.n()];
    for _ in 0..8 {
        cpu_step(&map, &sel, &wave, 0.25, &mut x, &r_a, &mut r_b, None);
        std::mem::swap(&mut r_a, &mut r_b);
    }
    assert_eq!(r_a[2], 0.0, "neuron 2 is a pruned type and must emit 0.0");
    assert_eq!(r_a[3], 0.0, "neuron 3 is a pruned type and must emit 0.0");
    assert!(
        r_a[0].abs() > 1e-6,
        "the driven neuron must be alive, otherwise the test proves nothing"
    );
    // The state behind the mask keeps integrating: pruning silences the OUTPUT, it does
    // not freeze the neuron. A future "pruned means dead" change has to face this line.
    assert!(
        x[2].abs() > 1e-6,
        "the pruned neuron still integrates its input; only r is masked"
    );
}

#[test]
fn an_unbound_channel_drives_nothing() {
    let map = FlyMap::synthetic();
    let layout = map.fly_layout();
    // No channel bound at all: every `in_bind` stays at -1.
    let block = block_of(layout, &[], [0, 1, 0, 1, 0, 1]);
    let sel = selection_from(&[block], layout);
    assert!(
        sel.group_to_input.iter().all(|&g| g == NONE),
        "no binding may leave a group driven"
    );

    let mut wave = vec![127i8; IN_CHANNELS];
    wave[0] = -128;
    let mut x = vec![0.0f32; map.n()];
    let mut r_a = vec![0.0f32; map.n()];
    let mut r_b = vec![0.0f32; map.n()];
    for _ in 0..20 {
        cpu_step(&map, &sel, &wave, 0.25, &mut x, &r_a, &mut r_b, None);
        std::mem::swap(&mut r_a, &mut r_b);
    }
    assert!(
        r_a.iter().all(|&v| v == 0.0),
        "with no input bound, a map born at x = 0 has nothing to integrate: {r_a:?}"
    );
    let cmd = cpu_read(&map, &sel, &r_a);
    assert_eq!(cmd[0], [0.0, 0.0, 0.0], "silent map, silent commands");
}

#[test]
fn a_command_is_the_difference_of_two_group_means() {
    let map = FlyMap::synthetic();
    let layout = map.fly_layout();
    // throttle = group 0 (neurons 4,5) minus group 1 (neurons 2,3);
    // steer  = group 1 minus group 0; brake = group 0 minus NONE (the unknown id 9).
    let block = block_of(layout, &[(0, 0)], [0, 1, 1, 0, 0, 9]);
    let sel = selection_from(&[block], layout);
    // Hand-written activations, so the expected value is arithmetic and not a re-run of
    // the model under test.
    let r = vec![0.1f32, -0.2, 0.3, 0.5, -0.4, 0.8];
    let m0 = (r[4] + r[5]) / 2.0;
    let m1 = (r[2] + r[3]) / 2.0;
    let cmd = cpu_read(&map, &sel, &r);
    assert!(
        (cmd[0][0] - (m0 - m1)).abs() < 1e-6,
        "throttle = mean0 - mean1"
    );
    assert!(
        (cmd[0][1] - (m1 - m0)).abs() < 1e-6,
        "steer = mean1 - mean0"
    );
    assert!(
        (cmd[0][2] - m0).abs() < 1e-6,
        "an out-of-range group id reads as NONE and contributes exactly 0"
    );
}

#[test]
fn the_state_is_being_minor_and_beings_do_not_leak_into_each_other() {
    let map = FlyMap::synthetic();
    let layout = map.fly_layout();
    let driven = block_of(layout, &[(0, 0)], [0, 1, 0, 1, 0, 1]);
    let mut silent = block_of(layout, &[(0, 0)], [0, 1, 0, 1, 0, 1]);
    // Being 1 has the whole map pruned: every r it produces must be 0, whatever being 0
    // is doing next to it in the same buffer.
    for t in 0..layout.n_types {
        silent[OFF_TYPE_MASK + t] = 0.0;
    }
    let sel = selection_from(&[driven, silent], layout);
    let b_n = 2;

    let mut wave = vec![0i8; b_n * IN_CHANNELS];
    wave[0] = 127; // being 0, channel 0
    wave[IN_CHANNELS] = 127; // being 1, channel 0
    let mut x = vec![0.0f32; map.n() * b_n];
    let mut r_a = vec![0.0f32; map.n() * b_n];
    let mut r_b = vec![0.0f32; map.n() * b_n];
    for _ in 0..10 {
        cpu_step(&map, &sel, &wave, 0.25, &mut x, &r_a, &mut r_b, None);
        std::mem::swap(&mut r_a, &mut r_b);
    }
    for j in 0..map.n() {
        assert_eq!(
            r_a[j * b_n + 1],
            0.0,
            "being 1 is fully pruned; neuron {j} of it must be 0"
        );
    }
    assert!(
        r_a[0].abs() > 1e-6,
        "being 0 must still be driven — index j * B + b, not b * n + j"
    );
    // Same map, same wave, one being: the value has to match the B = 2 run bit for bit,
    // which is what "being-minor" buys and what the kernel relies on.
    let sel1 = pack_selection(
        &[&block_of(layout, &[(0, 0)], [0, 1, 0, 1, 0, 1])],
        layout,
        1,
    );
    let mut x1 = vec![0.0f32; map.n()];
    let mut r1a = vec![0.0f32; map.n()];
    let mut r1b = vec![0.0f32; map.n()];
    for _ in 0..10 {
        cpu_step(
            &map,
            &sel1,
            &wave[..IN_CHANNELS],
            0.25,
            &mut x1,
            &r1a,
            &mut r1b,
            None,
        );
        std::mem::swap(&mut r1a, &mut r1b);
    }
    for j in 0..map.n() {
        assert_eq!(
            r_a[j * b_n],
            r1a[j],
            "neuron {j} of being 0 must not depend on how many beings share the buffer"
        );
    }
}

#[test]
fn when_two_channels_bind_the_same_group_the_higher_channel_wins() {
    let map = FlyMap::synthetic();
    let layout = map.fly_layout();
    let block = block_of(layout, &[(3, 0), (57, 0)], [0, 1, 0, 1, 0, 1]);
    let sel = selection_from(&[block], layout);
    assert_eq!(
        sel.group_to_input[0], 57,
        "the rule of §3 is that the higher channel wins the collision"
    );
}

#[test]
fn a_block_of_the_wrong_length_packs_as_inert_instead_of_garbage() {
    let map = FlyMap::synthetic();
    let layout = map.fly_layout();
    // What a genome saved before this change (or against another map) carries.
    let sel = pack_selection(&[&[]], layout, 1);
    assert!(sel.group_to_input.iter().all(|&g| g == NONE));
    assert!(sel.out_pair.iter().all(|&g| g == NONE));
    assert_eq!(sel.gain(0), 1.0);
    assert_eq!(sel.scale(0), 1.0);
    assert!(sel.type_mask.iter().all(|&m| m == 1.0));
    let r = vec![0.5f32; map.n()];
    assert_eq!(
        cpu_read(&map, &sel, &r)[0],
        [0.0, 0.0, 0.0],
        "an inert selection must command nothing"
    );
}

#[test]
fn a_newborn_block_has_the_layout_length_and_is_neutral_where_it_must_be() {
    let layout = FlyLayout {
        k_in: 267,
        k_out: 516,
        n_types: 8866,
    };
    let block = random_block(4242, layout);
    assert_eq!(block.len(), IN_CHANNELS + OUT_SLOTS + 2 + layout.n_types);
    assert_eq!(block[OFF_GAIN], 1.0, "gain is born at 1");
    assert_eq!(block[OFF_SCALE], 1.0, "scale is born at 1");
    assert_eq!(
        mask_density(&block, layout),
        1.0,
        "the whole mask is born on: generation 0 prunes nothing"
    );
    let bound = bound_channels(&block, layout).len();
    // 192 channels at p = 0.25 — the binomial 3σ band is 48 ± 18.
    assert!(
        (20..=80).contains(&bound),
        "{bound} bound channels is outside the 3σ band of a p = 0.25 draw"
    );
    assert!(
        bound_channels(&block, layout)
            .iter()
            .all(|&(_, g)| (g as usize) < layout.k_in),
        "a birth draw must never name a group the map does not have"
    );
    for s in 0..OUT_SLOTS {
        let g = block[OFF_OUT_PAIR + s];
        assert!(
            g >= 0.0 && (g as usize) < layout.k_out,
            "out slot {s} = {g}"
        );
    }
}

#[test]
fn the_pruning_operator_switches_types_off_and_back_on() {
    struct Always(f32);
    impl crate::gpu::evolution::GenoRng for Always {
        fn chance(&mut self) -> f32 {
            self.0
        }
        fn gauss(&mut self) -> f32 {
            0.0
        }
    }
    let layout = FlyLayout {
        k_in: 4,
        k_out: 2,
        n_types: 16,
    };
    let mut block = random_block(7, layout);
    // `chance` pinned at 0.0 makes every probability fire.
    mutate_block(&mut block, layout, 0.0, 1.0, 1.0, 0.1, &mut Always(0.0));
    assert_eq!(
        mask_density(&block, layout),
        0.0,
        "p_off = 1 must switch every type off"
    );
    mutate_block(&mut block, layout, 0.0, 1.0, 1.0, 0.1, &mut Always(0.0));
    assert_eq!(
        mask_density(&block, layout),
        1.0,
        "p_on = 1 must bring every type back"
    );
    assert_eq!(
        block[OFF_GAIN], 1.0,
        "gauss = 0 leaves the gain exactly as it was"
    );
}
