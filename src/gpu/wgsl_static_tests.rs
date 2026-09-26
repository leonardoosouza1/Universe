//! Static WGSL validation of every shader module the runtime creates, without a
//! GPU: `naga` parses and type-checks the same source string that reaches
//! `wgpu::Device::create_shader_module`. `cargo build` only checks that each
//! `include_str!` is a valid Rust `&str`.
//!
//! This lives in the library (not `tests/`) on purpose: `cargo test --test
//! wgsl_validation` links all 41 binaries (~0.7 GB each) and filled the system
//! disk on 2026-09-25, so that gate was never run after `brain.wgsl` changed.
//! `cargo test --lib wgsl_static` builds one test binary.
//!
//! Fragments that are only ever concatenated (`retina_geometry.wgsl`) are
//! validated through the same assembled constant the runtime uses
//! (`RETINA_SHADER`). The `physics_minimal/phase_*/power_*` fragments belong to
//! the `fisica_bisect` diagnostic binary and are assembled there; `physics.wgsl`
//! is not included by any module (historical three-pass predecessor).

fn validate(name: &str, source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("{name}: parse failed:\n{}", e.emit_to_string(source)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map(|_| ())
    .map_err(|e| format!("{name}: validation failed:\n{}", e.emit_to_string(source)))
}

/// Every (name, source) pair the runtime hands to `create_shader_module`.
fn runtime_shader_modules() -> Vec<(&'static str, &'static str)> {
    vec![
        ("brain.wgsl", include_str!("brain.wgsl")),
        ("cortex_sheet.wgsl", include_str!("cortex_sheet.wgsl")),
        (
            "cortex_sheet_hebb.wgsl",
            include_str!("cortex_sheet_hebb.wgsl"),
        ),
        (
            "cortex_sheet_motor.wgsl",
            include_str!("cortex_sheet_motor.wgsl"),
        ),
        ("fly_map/fly_map.wgsl", include_str!("fly_map/fly_map.wgsl")),
        ("fly_map/spiking.wgsl", include_str!("fly_map/spiking.wgsl")),
        ("organism.wgsl", include_str!("organism.wgsl")),
        ("organism_action.wgsl", include_str!("organism_action.wgsl")),
        (
            "organism_feedback.wgsl",
            include_str!("organism_feedback.wgsl"),
        ),
        (
            "organism_orientation.wgsl",
            include_str!("organism_orientation.wgsl"),
        ),
        (
            "organism_reproduction.wgsl",
            include_str!("organism_reproduction.wgsl"),
        ),
        (
            "organism_sensors.wgsl",
            include_str!("organism_sensors.wgsl"),
        ),
        (
            "organism_telemetry.wgsl",
            include_str!("organism_telemetry.wgsl"),
        ),
        ("perception.wgsl", include_str!("perception.wgsl")),
        (
            "perception_temporal.wgsl",
            include_str!("perception_temporal.wgsl"),
        ),
        (
            "perception_motion.wgsl",
            include_str!("perception_motion.wgsl"),
        ),
        ("physics_a.wgsl", include_str!("physics_a.wgsl")),
        ("physics_b.wgsl", include_str!("physics_b.wgsl")),
        ("physics_c.wgsl", include_str!("physics_c.wgsl")),
        (
            "RETINA_SHADER (retina_geometry + retina)",
            super::RETINA_SHADER,
        ),
        ("retina_digest.wgsl", include_str!("retina_digest.wgsl")),
        ("retina_metrics.wgsl", include_str!("retina_metrics.wgsl")),
        ("school.wgsl", include_str!("school.wgsl")),
        ("slow_brain.wgsl", include_str!("slow_brain.wgsl")),
        ("slow_gather.wgsl", include_str!("slow_gather.wgsl")),
        ("taxi.wgsl", include_str!("taxi.wgsl")),
        ("taxi_witness.wgsl", include_str!("taxi_witness.wgsl")),
        ("traffic.wgsl", include_str!("traffic.wgsl")),
        (
            "traffic_contacts.wgsl",
            include_str!("traffic_contacts.wgsl"),
        ),
    ]
}

#[test]
fn every_runtime_shader_module_parses_and_type_checks() {
    let failures: Vec<String> = runtime_shader_modules()
        .into_iter()
        .filter_map(|(name, source)| validate(name, source).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} shader module(s) fail static validation:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// A shader added to `src/gpu/` but not to the list above would be built by the
/// runtime and never validated here; this keeps the two in step.
#[test]
fn every_wgsl_file_is_either_validated_or_declared_a_fragment() {
    const FRAGMENTS_OR_UNUSED: &[&str] = &[
        "retina_geometry.wgsl", // concatenated into RETINA_SHADER
        "retina.wgsl",          // concatenated into RETINA_SHADER
        "physics.wgsl",         // not included by any module
        "physics_minimal.wgsl", // fisica_bisect fragments
        "physics_phase_integ.wgsl",
        "physics_phase_power.wgsl",
        "physics_phase_susp.wgsl",
        "physics_power_a.wgsl",
        "physics_power_b.wgsl",
        "physics_power_c.wgsl",
    ];
    let validated: Vec<&str> = runtime_shader_modules()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu");
    let mut missing = Vec::new();
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                continue;
            }
            let name = path
                .strip_prefix(&dir)
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            if !validated.contains(&name.as_str()) && !FRAGMENTS_OR_UNUSED.contains(&name.as_str())
            {
                missing.push(name);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "shaders neither validated nor declared fragments: {missing:?}"
    );
}

// ---------------------------------------------------------------------------
// Source-text canaries ported from `tests/wgsl_validation.rs` (archived at
// `docs/arquivo/wgsl_validation-superseded-20260925.rs.txt`). Each one guards a
// silent wrong answer that parse + type-check cannot see.
// ---------------------------------------------------------------------------

/// The native wave is packed as 48 u32 words, exactly 192 signed bytes per being.
/// Channel 192 is the first byte of the next being, not spare capacity.
#[test]
fn perception_temporal_never_writes_past_the_native_wave() {
    let source = include_str!("perception_temporal.wgsl");
    assert!(
        !source.contains("write_channel(being, 192u"),
        "channel 192 aliases channel 0 of the next being"
    );
}

/// `PerceptionGpu` dispatches one workgroup per being and eleven local threads
/// inside that workgroup. Every local thread must therefore index the same being;
/// using `global_invocation_id.x` silently assigns each sensor family to a
/// different member of the population.
#[test]
fn motion_workgroup_maps_every_local_thread_to_the_same_being() {
    let source = include_str!("perception_motion.wgsl");
    assert!(
        source.contains("@builtin(workgroup_id)"),
        "the motion shader must identify the being from workgroup_id"
    );
    assert!(
        source.contains("let being = workgroup.x;"),
        "all eleven local sensor threads must use their shared workgroup as the being index"
    );
    assert!(
        !source.contains("let being = gid.x;"),
        "global_invocation_id spreads one being's sensor threads across different beings"
    );
}

/// Channels 21..27 are legacy overwrite slots: the spatial pass writes wall
/// probes first, then the motion pass replaces those bytes with retinal deltas.
/// `atomicOr` alone merges the two unrelated measurements bit by bit.
#[test]
fn motion_rows_replace_legacy_spatial_slots_instead_of_merging_them() {
    let source = include_str!("perception_motion.wgsl");
    assert!(
        source.contains("fn overwrite_channel"),
        "motion-over-wall slots need an explicit byte replacement primitive"
    );
    assert!(
        source.contains("overwrite_channel(being, 21u + row, motion);"),
        "rows 0..6 must replace channels 21..27 after the spatial pass"
    );
    assert!(
        !source
            .lines()
            .any(|line| line.trim() == "write_channel(being, 21u + row, motion);"),
        "plain atomicOr corrupts channels 21..27 by merging wall and motion bits"
    );
}

/// The eye extension of `fly_map.wgsl` is a conditional inside `integrate()`, easy to
/// lose in an edit (docs/FLY-MAP-BRAIN.md "Eye binding").
#[test]
fn fly_map_keeps_the_eye_binding_drive() {
    let source = include_str!("fly_map/fly_map.wgsl");
    assert!(
        source.contains("eye_pixel_drive") && source.contains("pixel_of_neuron"),
        "the eye-binding drive (docs/FLY-MAP-BRAIN.md \"Eye binding\") is missing from fly_map.wgsl"
    );
}

/// `--retina-beings` (docs/RETINA-BEINGS.md §2): the occlusion path and the
/// half-cell rounding that matches Rust `f32::round` must survive edits.
#[test]
fn retina_keeps_being_occlusion_and_round_away_from_zero() {
    let source = super::RETINA_SHADER;
    assert!(
        source.contains("BEING_MATERIAL") && source.contains("beings_on"),
        "the being-occlusion path (docs/RETINA-BEINGS.md §2) is missing from retina.wgsl"
    );
    assert!(
        source.contains("fn round_away_from_zero")
            && source.contains("round_away_from_zero(wx / params.cell_size)"),
        "retina.wgsl must make its grid rounding match Rust f32::round at half-cells"
    );
}

/// The four copies of `legacy_encoding` must agree on the Being branch: the Rust
/// reference `gpu::legacy_encoding` and WGSL copies in `fly_map.wgsl::eye_pixel_drive`,
/// `perception.wgsl` and `retina_digest.wgsl`. When `--retina-beings` shipped, two of
/// the four had the Being branch and two silently fell through to Hole.
#[test]
fn every_legacy_encoding_copy_handles_the_being_material() {
    for (name, src) in [
        ("perception.wgsl", include_str!("perception.wgsl")),
        ("retina_digest.wgsl", include_str!("retina_digest.wgsl")),
        ("fly_map.wgsl", include_str!("fly_map/fly_map.wgsl")),
        ("spiking.wgsl", include_str!("fly_map/spiking.wgsl")),
    ] {
        assert!(
            src.contains("else if mat == 3u") && src.contains("-29"),
            "{name} must encode Being separately from Hole as -29"
        );
    }
    assert_eq!(
        super::legacy_encoding(super::Pixel {
            dist_m: 7.0,
            material: super::BEING_MATERIAL,
        }),
        -29,
        "the Rust reference must encode Being as -29"
    );
}

/// WGSL cannot import Rust constants, so the brain and perception kernels each
/// redeclare the wave width and layer sizes as literals. `src/CLAUDE.md` records
/// the stale-comment incident this guards against (`223 x 96` when the wave was
/// already 192). Whitespace inside the declaration is irrelevant to WGSL, so the
/// comparison collapses runs of spaces.
#[test]
fn wgsl_dimension_literals_match_config() {
    use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
    assert_eq!(
        (SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS),
        (192, 96, 48),
        "config.rs dimensions changed: update every WGSL literal listed in this test"
    );
    let wave_u32s = SENSORY_WAVE_SIZE.div_ceil(4);
    let expect: &[(&str, &str, &[String])] = &[
        (
            "brain.wgsl",
            include_str!("brain.wgsl"),
            &[
                format!("const WAVE: u32 = {SENSORY_WAVE_SIZE}u;"),
                format!("const WAVE_U32S: u32 = {wave_u32s}u;"),
                format!("const L1_OUT: u32 = {L1_NEURONS}u;"),
                format!("const L2_OUT: u32 = {L2_NEURONS}u;"),
                format!(
                    "const NEURON_META_PER_BEING: u32 = {}u;",
                    super::brain::NEURON_META_PER_BEING
                ),
                format!(
                    "const NEURON_MEM_PER_BEING: u32 = {}u;",
                    super::brain::NEURON_MEM_PER_BEING
                ),
            ],
        ),
        (
            "perception.wgsl",
            include_str!("perception.wgsl"),
            &[
                format!("const WAVE_SIZE: u32 = {SENSORY_WAVE_SIZE}u;"),
                format!("const WAVE_U32S: u32 = {wave_u32s}u;"),
            ],
        ),
        (
            "perception_motion.wgsl",
            include_str!("perception_motion.wgsl"),
            &[format!("const WAVE_U32S: u32 = {wave_u32s}u;")],
        ),
        (
            "perception_temporal.wgsl",
            include_str!("perception_temporal.wgsl"),
            &[format!("const WAVE_U32S: u32 = {wave_u32s}u;")],
        ),
        (
            "retina_digest.wgsl",
            include_str!("retina_digest.wgsl"),
            &[format!("const WAVE_U32S: u32 = {wave_u32s}u;")],
        ),
        (
            "slow_gather.wgsl",
            include_str!("slow_gather.wgsl"),
            &[format!("const WAVE_U32S: u32 = {wave_u32s}u;")],
        ),
    ];
    for (name, source, literals) in expect {
        let collapsed: String = source.split_whitespace().collect::<Vec<_>>().join(" ");
        for literal in literals.iter() {
            assert!(
                collapsed.contains(literal.as_str()),
                "{name} no longer declares `{literal}`; the Rust constant it mirrors changed or the shader was edited"
            );
        }
    }
}
