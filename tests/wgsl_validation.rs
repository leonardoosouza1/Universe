//! Static validation of the WGSL shaders modified in the CHANNELS PHASE
//! (2026-08-14), WITHOUT a GPU — `naga::front::wgsl::parse_str` +
//! `naga::valid::Validator` run entirely on the CPU (parse + type-check), the
//! same verification `wgpu::Device::create_shader_module` would do internally,
//! but without needing a real device.
//!
//! # Why this test exists
//!
//! The project's `.wgsl` kernels are embedded via `include_str!` as plain text
//! — `cargo build`/`cargo test` NEVER check the WGSL syntax or types, only that
//! the string compiles as a Rust `&str`. A WGSL syntax error (missing
//! parenthesis, wrong type, duplicated channel name) would only show up at
//! RUNTIME, the first time `device.create_shader_module()` ran — i.e. only on a
//! real GPU. The project rule in that session was "run nothing on the GPU:
//! 2 curriculum batteries are occupying it", so this test is the only way to
//! know, with `cargo test`, whether the edited shaders
//! (`perception_temporal.wgsl`, `perception_motion.wgsl` — ch101, 109, 114-117,
//! CHANNELS PHASE) at least PARSE and TYPE-CHECK before anyone runs
//! `gpu_evolve` for real.
//!
//! What this does NOT prove: that the LOGIC is right (channel semantics,
//! calibrated normalizers) — only that the shader is valid WGSL. Behavior
//! validation (σ_temporal/σ_entre via `--auditar-visao`) remains pending,
//! documented in the shaders' own comments.

fn validar_wgsl(nome: &str, fonte: &str) {
    let module = naga::front::wgsl::parse_str(fonte)
        .unwrap_or_else(|e| panic!("{nome}: WGSL PARSE failed:\n{}", e.emit_to_string(fonte)));

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).unwrap_or_else(|e| {
        panic!(
            "{nome}: WGSL VALIDATION (type-check) failed:\n{}",
            e.emit_to_string(fonte)
        )
    });
}

/// Edited in the CHANNELS PHASE: ch101 (curvature 260m, reuses `curv_at_d`) and
/// ch109 (real derivative of the racing line, previously hardcoded 0.0).
#[test]
fn perception_temporal_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/perception_temporal.wgsl");
    validar_wgsl("perception_temporal.wgsl", fonte);
}

/// Edited in the CHANNELS PHASE: ch114-117 (hemispheric approach + fovea), new
/// `write_channel_signed` function, `workgroup_size` from 10 to 11 threads.
#[test]
fn perception_motion_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/perception_motion.wgsl");
    validar_wgsl("perception_motion.wgsl", fonte);
}

/// CHANNELS PHASE (2026-08-14): initially comments only ("NOT ported" header
/// updated to reflect ch114-117 leaving the "motor.rs only" list).
///
/// SENSORS PHASE, step 1 (2026-08-14, substantive change): FOV of the
/// wall/curvature sweep (BLOCK C/D/E) and ranges/lookaheads (BLOCK J/K) stopped
/// being compile-time `const` and became heritable per-being genes, read from a new
/// buffer (`sensors`, binding 7 of `layout_spatial`) — see `gpu::evolution::N_SENSOR_GENES`/
/// `PerceptionGpu::upload_sensors`. Structural change (new binding, bind group
/// layout with 1 more entry, per-being read at the start of `main()`) — exactly the
/// kind of edit where a syntax/type error would only show up on a real GPU.
#[test]
fn perception_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/perception.wgsl");
    validar_wgsl("perception.wgsl", fonte);
}

/// Edited in P1.5 (`docs/PLANO_EVOLUCAO.md`): `Params` gained `out_gain` +
/// `norm_l2_by_std` (+ padding to keep the uniform a multiple of 16 bytes),
/// STAGE 2 was restructured to separate the L2 matmul (kept in LDS, new
/// `w_l2r[48]`) from the gain/fatigue/tanh — with a new `workgroupBarrier()` in
/// uniform control flow in between — to allow the optional standard-deviation
/// normalization. Structural change (new barrier, new LDS array, nested `if`
/// computing `l2_std`), not just a literal swap — exactly the kind of edit where
/// a syntax/type error would only show up on a real GPU; this test catches it
/// without needing one.
#[test]
fn brain_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/brain.wgsl");
    validar_wgsl("brain.wgsl", fonte);
}

/// Edited in P1.8 (docs/PLANO_EVOLUCAO.md, critical finding "ch192-207 aliased
/// into the NEXT being"): the loop that wrote `write_channel(being, 192u+i, val)`
/// into `output` (shared buffer, no room for the 16 extra channels — corrupted
/// channels 0-15 of the following being via atomicOr) now writes into
/// `extra_out[being*EXTRA_STRIDE + 3 + i]` (its own per-being space). Structural
/// change (new `EXTRA_STRIDE` const, three `extra_out` writes re-indexed),
/// validated on the real GPU by
/// `retina_digest_parity_test::retina_digest_bate_com_cpu_ao_longo_de_varios_ticks`
/// (`--ignored`) — this test here is the no-GPU safety net, same pattern as the
/// others in this suite.
#[test]
fn retina_digest_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/retina_digest.wgsl");
    validar_wgsl("retina_digest.wgsl", fonte);
}

/// FLY MAP (docs/FLY-MAP-BRAIN.md §4): two new entry points, `fly_step` and `fly_read`,
/// in one module with 19 bindings (17 + the eye binding's `pixel_of_neuron` and `retina`,
/// `docs/FLY-MAP-BRAIN.md` "Eye binding") — the largest binding set in the tree, and the
/// first shader here to index a private array (`acc[64]`) with a runtime value inside a
/// nested loop. Every one of those is a construct whose type error would only appear at
/// `create_shader_module`, i.e. on a real GPU, in the middle of a run.
#[test]
fn fly_map_wgsl_is_valid() {
    let fonte = include_str!("../src/gpu/fly_map/fly_map.wgsl");
    validar_wgsl("fly_map.wgsl", fonte);
    // The eye extension is easy to lose in a future edit of `integrate()` (it is a
    // conditional inside an already-dense function, not a separate call site) — a no-GPU
    // canary next to the parse check, cheaper than waiting for `fly_map_parity` to notice.
    assert!(
        fonte.contains("eye_pixel_drive") && fonte.contains("pixel_of_neuron"),
        "the eye-binding drive (docs/FLY-MAP-BRAIN.md \"Eye binding\") is missing from \
         fly_map.wgsl but this test's doc comment still describes it"
    );
}

#[test]
fn fly_spiking_wgsl_is_valid() {
    let fonte = include_str!("../src/gpu/fly_map/spiking.wgsl");
    validar_wgsl("fly_map/spiking.wgsl", fonte);
}
