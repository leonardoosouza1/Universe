//! BODY BLOCK — packing tests (no GPU) and a CPU×GPU parity test (`#[ignore]`).
//!
//! # What is under test
//!
//! The body block is the single transport that carries CPU-born signals — the
//! being's own body (fuel, tyre thermal, engine temp) and the hostile world
//! (energy station, predator) — into the sensory wave. It has three sides that must
//! agree, and nothing in the build system checks two of the three:
//!
//! 1. the registry, `perception::channels::BODY_SLOTS` (slot → channel);
//! 2. the host packer, `PerceptionGpu::update_body_channels` (slot → byte offset);
//! 3. the kernel, `perception_temporal.wgsl` (slot → `write_channel`), plus
//!    `perception_motion.wgsl` going silent on the five channels that change owner.
//!
//! `cargo build` only proves the `.wgsl` is a valid Rust `&str`, and
//! `wgsl_validation` only proves it parses and type-checks. Neither can see that
//! the shader reads slot 5 for a channel the registry assigns to slot 4 — that
//! disagreement produces a plausible number, no error and no panic. So the
//! agreement is enforced here: by text assertions for the parts a CPU test can
//! reach, and by the `#[ignore]` parity test for the part only a GPU can answer.
//!
//! # Why the parity test is `#[ignore]`
//!
//! Same reason and same shape as `retina_digest_parity_test.rs` (the model this
//! file follows): this crate does not open a `wgpu::Device` inside a normal
//! `cargo test`, so machines and CI without a GPU keep passing. Run it explicitly:
//!
//! ```text
//! HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
//!   cargo test --release --lib -- --ignored body_block
//! ```

use super::{
    body_slot_byte_offset, encode_body_signed, encode_body_unsigned, PerceptionGpu, PosePerception,
    BODY_SLOT_BASE, CURVE_SLOT_BASE, CURVE_SLOT_COUNT, TEMPORAL_STRIDE,
};
use crate::config::iara::brain::SENSORY_WAVE_SIZE;
use crate::config::iara::vision::{RETINA_COLS, RETINA_TOTAL};
use crate::perception::channels::{BODY_K, BODY_SLOTS};

/// Source of the kernel that reads the block, for the text-agreement tests.
const TEMPORAL_WGSL: &str = include_str!("perception_temporal.wgsl");
/// Source of the kernel that must go silent on the five handed-off channels.
const MOTION_WGSL: &str = include_str!("perception_motion.wgsl");

/// Strips every ASCII whitespace character so an assertion is about the CODE, not
/// about how it happens to be aligned. Without this, `write_channel(being,  98u,`
/// (padded to line up with the three-digit channels) would fail a search for
/// `write_channel(being, 98u,` — a test failing on formatting is a test that gets
/// deleted the first time someone runs `rustfmt` on the shader comments.
fn squeeze(src: &str) -> String {
    src.chars().filter(|c| !c.is_whitespace()).collect()
}

// ─── Packing (no GPU) ──────────────────────────────────────────────────────────

/// The curve block immediately follows the body block and closes the stride.
///
/// If `BODY_SLOT_BASE + BODY_K` were smaller than the stride there would be dead
/// slots (harmless but undocumented); if it were larger, being `i`'s block would
/// write over being `i+1`'s state with no error at all. That second case is the
/// failure `gpu/CLAUDE.md` records for `retina_digest`, which wrote past its region
/// and landed on the next being's channels 0-15 — hence the locked stride there
/// (`EXTRA_OUT_STRIDE = 19`) and hence this test.
#[test]
fn body_and_curve_blocks_close_the_stride() {
    assert_eq!(
        BODY_SLOT_BASE + BODY_K,
        CURVE_SLOT_BASE,
        "curve slots must start immediately after the body block"
    );
    assert_eq!(
        CURVE_SLOT_BASE + CURVE_SLOT_COUNT,
        TEMPORAL_STRIDE,
        "curve slots must end exactly at the temporal stride"
    );
}

/// The block was APPENDED: it starts after slot 57, the last slot of
/// `update_extra2_temporal`. Any overlap would silently overwrite a channel that is
/// already measured and in use (ch87, ch99-100, ch121-122, ch124-127).
#[test]
fn body_block_does_not_invade_slots_already_in_use() {
    assert!(
        BODY_SLOT_BASE >= 58,
        "BODY_SLOT_BASE={BODY_SLOT_BASE} overlaps slots 49-57 of update_extra2_temporal"
    );
}

/// The byte offset is `(being * stride + base) * 4`, and consecutive beings never
/// overlap. Checked over several beings because an off-by-one in the base only
/// shows up as corruption from being 1 onwards — being 0 reads correctly either
/// way, which is exactly how the `PhysicsStateGpu` stride bug stayed invisible.
#[test]
fn block_offset_matches_and_never_overlaps_the_next_being() {
    for being in 0..8usize {
        let want = (being * TEMPORAL_STRIDE + BODY_SLOT_BASE) as u64 * 4;
        assert_eq!(
            body_slot_byte_offset(being),
            want,
            "byte offset of being {being}"
        );
        // This being's block ends before the next being's region begins.
        let block_end = body_slot_byte_offset(being) + (BODY_K * 4) as u64;
        let next_being_start = ((being + 1) * TEMPORAL_STRIDE) as u64 * 4;
        assert!(
            block_end <= next_being_start,
            "being {being}: the block ends at {block_end} and being {}'s region starts at \
             {next_being_start} — the write would leak into the neighbour",
            being + 1
        );
    }
    // Numeric anchors: if the stride or the base moves, these move with it, and
    // whoever moves them is forced to look at them.
    assert_eq!(body_slot_byte_offset(0), 58 * 4);
    assert_eq!(body_slot_byte_offset(1), (TEMPORAL_STRIDE + 58) as u64 * 4);
}

/// The host encoders are the twins of the shader's `clamp(v * 127.0, ...)`.
/// Signed saturates at ±127 (not −128 — the block has no `{0,-128}` flag), unsigned
/// floors at 0, and a value passed outside its declared range clamps instead of
/// wrapping. That last one matters: passing metres where [0,1] was asked for
/// produces a channel pinned at 127, which is a visible canary, not garbage.
#[test]
fn block_encoding_saturates_instead_of_wrapping() {
    assert_eq!(encode_body_signed(0.0), 0);
    assert_eq!(encode_body_signed(1.0), 127);
    assert_eq!(encode_body_signed(-1.0), -127);
    assert_eq!(
        encode_body_signed(2.0),
        127,
        "above the range it saturates, it does not wrap"
    );
    assert_eq!(encode_body_signed(-9.0), -127);
    assert_eq!(encode_body_signed(0.5), 63, "63.5 truncates to 63");

    assert_eq!(encode_body_unsigned(0.0), 0);
    assert_eq!(encode_body_unsigned(1.0), 127);
    assert_eq!(
        encode_body_unsigned(-0.3),
        0,
        "a negative on an unsigned channel becomes 0, never 255/-1"
    );
    assert_eq!(encode_body_unsigned(300.0), 127);
}

// ─── Agreement between registry and kernel (no GPU) ────────────────────────────

/// WGSL cannot import a Rust `const`, so `perception_temporal.wgsl` writes the
/// stride and the block base by hand. This test is the only thing standing between
/// a stride edit on one side and silent corruption on the other.
#[test]
fn shader_consts_match_the_host() {
    let src = squeeze(TEMPORAL_WGSL);
    assert!(
        src.contains(&format!("constTEMPORAL_STRIDE:u32={TEMPORAL_STRIDE}u;")),
        "perception_temporal.wgsl does not declare TEMPORAL_STRIDE={TEMPORAL_STRIDE}u"
    );
    assert!(
        src.contains(&format!("constBODY_SLOT_BASE:u32={BODY_SLOT_BASE}u;")),
        "perception_temporal.wgsl does not declare BODY_SLOT_BASE={BODY_SLOT_BASE}u"
    );
}

/// The uniform is 32 bytes on the host and all THREE kernels declare the same
/// fields, in the same order.
///
/// `body_on` was appended to a struct that was 16 bytes. WGSL rounds a uniform
/// struct up to its 16-byte alignment, so 5 f32 on the host (20 bytes) against a
/// shader type WGSL sizes as 32 is a binding smaller than the shader demands. And
/// the reverse — one kernel declaring the field and another not — is worse: the
/// buffer is shared by the three pipelines, so a field at a different offset in one
/// of them reads a neighbouring field's bits with no error at all. This is the same
/// class of failure `gpu/CLAUDE.md` records for `PhysicsStateGpu` (360 bytes on the
/// host, 368 in the shader's array: "being 0 still reads correctly while every being
/// after it reads shifted garbage").
#[test]
fn uniform_is_32_bytes_and_identical_in_the_three_kernels() {
    assert_eq!(
        std::mem::size_of::<super::ParamsGpu>(),
        32,
        "ParamsGpu must stay a multiple of 16 bytes (WGSL uniform alignment)"
    );
    let fields = [
        "grid_size:u32,",
        "n_beings:u32,",
        "cell_size:f32,",
        "perif_real:f32,",
        "body_on:f32,",
        "_pad0:f32,",
        "_pad1:f32,",
        "_pad2:f32,",
    ];
    for (name, source) in [
        ("perception.wgsl", include_str!("perception.wgsl")),
        ("perception_temporal.wgsl", TEMPORAL_WGSL),
        ("perception_motion.wgsl", MOTION_WGSL),
    ] {
        let src = squeeze(source);
        // The fields must appear IN ORDER — a struct with the right fields in the
        // wrong order has the right size and the wrong offsets.
        let mut cursor = 0usize;
        for field in fields {
            let found = src[cursor..].find(field).unwrap_or_else(|| {
                panic!("{name}: uniform Params is missing field `{field}` (or has it out of order)")
            });
            cursor += found + field.len();
        }
    }
}

/// Every row of the registry has to appear in the kernel as a read of ITS slot and
/// a write to ITS channel. A shader reading slot 5 for a channel the registry
/// assigns to slot 4 compiles, validates, runs, and reports a plausible number.
#[test]
fn shader_reads_every_slot_and_writes_every_channel_of_the_registry() {
    let src = squeeze(TEMPORAL_WGSL);
    for &(slot, ch, name, _) in BODY_SLOTS {
        assert!(
            src.contains(&format!("bb+{slot}u")),
            "perception_temporal.wgsl never reads slot {slot} ({name}) — ch{ch} would be mute"
        );
        assert!(
            src.contains(&format!("write_channel(being,{ch}u,")),
            "perception_temporal.wgsl never writes ch{ch} ({name}, slot {slot})"
        );
    }
}

/// The five channels that change owner must be guarded on BOTH sides.
///
/// The motion kernel is the side that can only fail silently: if a guard is dropped
/// there, motion and temporal both `atomicOr` the same byte and the result is the OR
/// of two values — a number in range, with no error. There are four guards (ch98,
/// ch114, ch115, and the thread that owns ch116/117), all reading the same uniform
/// field.
#[test]
fn motion_goes_silent_on_the_five_handed_off_channels() {
    let src = squeeze(MOTION_WGSL);
    let guards = src.matches("params.body_on<=0.5").count();
    assert!(
        guards >= 4,
        "perception_motion.wgsl has {guards} body_on guards, expected >= 4 \
         (ch98, ch114, ch115, and the thread that owns ch116/117)"
    );
    // And the handoff only covers the block's channels: ch21-27 and ch83/84 keep
    // coming out of motion in both flag states, with no guard at all.
    for ch in [83u32, 84u32] {
        assert!(
            src.contains(&format!("write_channel(being,{ch}u,")),
            "ch{ch} is not part of the handoff and must still be written by motion"
        );
    }
}

// ─── CPU × GPU parity (`#[ignore]`) ────────────────────────────────────────────

const N_BEINGS: usize = 3;

/// Deterministic synthetic retina, varying with `being` so the motion channels are
/// not all identical — a test where every being sees the same thing cannot tell a
/// per-being stride bug from correct code.
fn synthetic_retina(being: usize) -> Vec<u32> {
    (0..RETINA_TOTAL)
        .map(|j| {
            let h = (j as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add((being as u32).wrapping_mul(7_919))
                .wrapping_mul(0x9E37_79B1);
            let dist_mm = h % 60_000; // 0..60 m
            let material = if (j % RETINA_COLS) >= 42 && (j % RETINA_COLS) < 84 {
                2u32 // wall in the central columns
            } else {
                1u32
            };
            (dist_mm << 8) | material
        })
        .collect()
}

/// Body values for being `i` — DIFFERENT per being and per slot, on purpose. If the
/// packer had an off-by-one in the stride, being 1 would read being 0's numbers and
/// a test with equal values everywhere would pass anyway.
fn body_values(being: usize) -> [f32; BODY_K] {
    let b = being as f32;
    [
        -0.75 + 0.25 * b, // slot 0, signed
        0.10 + 0.20 * b,  // slot 1
        0.90 - 0.30 * b,  // slot 2
        0.33 + 0.11 * b,  // slot 3
        0.60 - 0.40 * b,  // slot 4, signed
        0.05 + 0.15 * b,  // slot 5
        0.70 + 0.05 * b,  // slot 6
        0.45 + 0.05 * b,  // slot 7
    ]
}

/// Slots whose channel is signed ([-1,1] host range) — the two bearings.
fn slot_is_signed(slot: usize) -> bool {
    slot == 0 || slot == 4
}

async fn open_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("body-block-parity-test"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .ok()
}

/// Runs ONE tick of the full perception chain (spatial → temporal → motion) on a
/// fresh `PerceptionGpu` and returns the sensory wave of every being.
///
/// A fresh context per arm is deliberate: `buf_temporal` and `buf_retina_prev` are
/// resident between ticks, so running both arms on one context would compare the
/// flag against a state the first arm had already moved.
fn run_one_tick(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world: &crate::world::VoxelWorld,
    buf_retina: &wgpu::Buffer,
    poses: &[PosePerception],
    body_on: bool,
) -> Vec<[i8; SENSORY_WAVE_SIZE]> {
    let percept = PerceptionGpu::new(device, queue, world, N_BEINGS)
        .expect("PerceptionGpu::new failed with a GPU available");

    super::BODY_CHANNELS_ON.store(body_on, std::sync::atomic::Ordering::Relaxed);
    if body_on {
        let vals: Vec<[f32; BODY_K]> = (0..N_BEINGS).map(body_values).collect();
        percept.update_body_channels(queue, &vals);
    }

    let mut enc = device.create_command_encoder(&Default::default());
    percept.copy_retina_prev(&mut enc, buf_retina);
    let bytes = percept.bind_and_dispatch(&mut enc, device, queue, poses, buf_retina);
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
    let wave = percept.read_after_submit(device, N_BEINGS, bytes);

    // Leave the global as it was found — a static that survives a test is how a
    // later test in the same process inherits a flag nobody set for it.
    super::BODY_CHANNELS_ON.store(false, std::sync::atomic::Ordering::Relaxed);
    wave
}

/// Proves the three things the CPU tests above cannot:
///
/// 1. with the flag ON, every channel of the block carries exactly the value the
///    host packed for THAT being (per-being values, so a stride bug fails here);
/// 2. with the flag OFF the block is invisible — and the channels that are NOT part
///    of the handoff (ch21-27, ch83-84) come out bit-identical in both arms, so the
///    ownership switch disturbed nothing around it;
/// 3. the handed-off channels actually CHANGE between the arms, which is the "same
///    number where it should differ" canary (`CLAUDE.md` rule 2) applied on purpose:
///    if flag ON and flag OFF agreed on ch114-117, the block would be doing nothing.
#[test]
#[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored body_block"]
fn body_block_reaches_the_right_channel_of_every_being() {
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[body_block_parity] no GPU available — test skipped");
        return;
    };

    let world = crate::world::VoxelWorld::new(0, 28.0);
    let mut retina_data = Vec::with_capacity(N_BEINGS * RETINA_TOTAL);
    for being in 0..N_BEINGS {
        retina_data.extend_from_slice(&synthetic_retina(being));
    }
    let buf_retina = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("body-test-retina"),
        size: (N_BEINGS * RETINA_TOTAL * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf_retina, 0, bytemuck::cast_slice(&retina_data));

    let poses: Vec<PosePerception> = (0..N_BEINGS)
        .map(|i| PosePerception {
            pos_x: world.start_pos.x + i as f32 * 4.0,
            pos_y: world.start_pos.y,
            pos_z: world.start_pos.z,
            phase: 0.0,
            fwd_x: 1.0,
            fwd_z: 0.0,
            rgt_x: 0.0,
            rgt_z: 1.0,
        })
        .collect();

    let off = run_one_tick(&device, &queue, &world, &buf_retina, &poses, false);
    let on = run_one_tick(&device, &queue, &world, &buf_retina, &poses, true);

    // (1) Every channel of the block carries THAT being's value.
    for being in 0..N_BEINGS {
        let vals = body_values(being);
        for &(slot, ch, name, _) in BODY_SLOTS {
            let want = if slot_is_signed(slot) {
                encode_body_signed(vals[slot])
            } else {
                encode_body_unsigned(vals[slot])
            };
            let got = on[being][ch];
            assert!(
                (got as i32 - want as i32).abs() <= 1,
                "being {being} ch{ch} ({name}, slot {slot}): GPU={got} host={want} \
                 (injected value {})",
                vals[slot]
            );
        }
    }

    // (2) What is not part of the handoff comes out identical in both arms.
    for being in 0..N_BEINGS {
        for ch in [21usize, 22, 23, 24, 25, 26, 27, 83, 84] {
            assert_eq!(
                off[being][ch], on[being][ch],
                "being {being} ch{ch} is not in the block and moved with the flag — the \
                 handoff leaked into a neighbouring channel"
            );
        }
    }

    // (3) The handed-off channels DO change. Equal numbers here = block off (rule 2).
    let changed = BODY_SLOTS
        .iter()
        .any(|&(_, ch, _, _)| (0..N_BEINGS).any(|b| off[b][ch] != on[b][ch]));
    assert!(
        changed,
        "no channel of the block moved between flag off and flag on — the mechanism is \
         not wired (same class of bug as fuel-auto: mute, no error)"
    );
}
