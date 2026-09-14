//! The `flymap` v2/v3 trailing blocks, the "Eye binding" extension of §2 that reads one
//! of them, and the "Reflex readout" (`--fly-readout reflex`) that reads another.
//!
//! Split out of `fly_map.rs` for the 600-line cap (`docs/STYLE.md` §2), along the seam a
//! reader already perceives: `fly_map.rs` is v1 (the map and the rate model as they were
//! before this change), this file is everything v2/v3 added. Re-exported by `fly_map`, so
//! callers see one module.
//!
//! Two of the four trailing blocks — the presynaptic CSR and the no-refractory bitmap —
//! are parsed and stored here but read by NOTHING in this file: `docs/FLY-MAP-BRAIN.md`
//! §1 is explicit that they exist for a future spiking kernel, not the rate path this
//! task implements. The pixel-binding block feeds the dynamics, through [`eye_drive`].
//! The reflex-readout block (§1 v3, "Reflex readout") is read at the HOST level, not
//! inside `cpu_step`/`fly_step`: it overwrites `FlySelection::out_pair` before upload
//! ([`apply_reflex_readout`], `fly_map_selection.rs`), so the kernel that already computes
//! `mean(plus) - mean(minus)` over `out_pair` needed no edit to serve it.

use std::io::{Read, Write};

use super::*;

/// Bit 0 of the v2 header's `flags` word: the presynaptic CSR trails the v1 body.
pub const FLAG_PRESYNAPTIC_CSR: u32 = 1 << 0;
/// Bit 1: the pixel-binding block (`pixel_of_neuron`) trails it.
pub const FLAG_PIXEL_BINDING: u32 = 1 << 1;
/// Bit 2: the no-refractory bitmap trails it.
pub const FLAG_NO_REFRACTORY: u32 = 1 << 2;
/// Bit 3: the reflex-readout block (`reflex_out: u32[OUT_SLOTS]`) trails it — v3,
/// "Reflex readout" (`docs/FLY-MAP-BRAIN.md` §1). The last of the four trailing blocks,
/// always written after the no-refractory bitmap regardless of which other bits are set.
/// Its six entries are `(plus, minus)` pairs for throttle/steer/brake, the exact shape
/// `FlySelection::out_pair` already carries per being — see [`apply_reflex_readout`].
pub const FLAG_REFLEX_READOUT: u32 = 1 << 3;

/// The connectome read the other way round: `indptr[j]..indptr[j + 1]` are the edges for
/// which neuron `j` is the PRESYNAPTIC side (`indices[e]` the postsynaptic neuron,
/// `weights[e]` the same signed count the postsynaptic CSR carries for that edge).
///
/// Nothing in this crate reads it today. A spike has to walk OUT of a neuron the instant
/// it fires, which the postsynaptic CSR (indexed by the RECEIVING neuron) cannot do
/// without a full scan of `m` edges; this is here so the spiking kernel does not have to
/// re-derive it from 15.1 M edges at load time.
#[derive(Clone, Debug, Default)]
pub struct PresynapticCsr {
    /// `n + 1` entries.
    pub indptr: Vec<u32>,
    /// `m` entries: postsynaptic neuron of each edge.
    pub indices: Vec<u32>,
    /// `m` entries: the signed synapse count, duplicated from the postsynaptic CSR rather
    /// than cross-referenced — the two CSRs need no shared index to be read independently.
    pub weights: Vec<f32>,
}

/// One tick's rendered retina, in exactly the layout `fly_step` reads it on the GPU:
///
/// * `retina` is `GpuCtx::buf_retina_out` (`src/bin/gpu_evolve/context.rs:363-367`),
///   BEING-MAJOR — `retina[being * retina_stride + pixel]` — packed `(mm << 8) | material`
///   per pixel exactly as `src/gpu/retina.wgsl:159-160` writes it (`mm` = millimetres,
///   `material` = 0 Grass / 1 Road / 2 Wall / 3 Hole, `src/gpu/perception.wgsl:275-277`'s
///   `pixel_mat`).
/// * `retina_stride` is pixels per being; in production this is
///   `RETINA_TOTAL = 1764` (`src/config.rs`, mirrored as a literal at
///   `src/gpu/retina.wgsl:53`) — [`crate::gpu::fly_map::FlyMapGpu::dispatch`] always passes
///   that value, read from `config.rs` and not re-declared here (`docs/STYLE.md` §4).
/// * `pixel_of_neuron` is the map's own table ([`crate::fly_map::FlyMap::pixel_of_neuron`]),
///   passed separately because the CPU reference is exercised against maps that may or may
///   not carry the block.
pub struct EyeFrame<'a> {
    pub pixel_of_neuron: &'a [u32],
    pub retina: &'a [u32],
    pub retina_stride: usize,
}

/// The eye term of §2's `u`, for one (being, neuron) pair — `None` when this neuron is not
/// eye-driven (no block, or `pixel_of_neuron[neuron] == NONE`) or when the indices fall
/// outside what `frame` carries (a defensive `None`, not a panic: the caller falls back to
/// the genome binding exactly as it would for an absent block).
///
/// `docs/FLY-MAP-BRAIN.md` "Eye binding": `u_eye = gain · (1 - brightness)`, `brightness =
/// clamp(pixel_value / 127, 0, 1)`, where `pixel_value` is the SAME signed byte the 192
/// sensory channels are built from — [`crate::gpu::legacy_encoding`]
/// (`src/gpu/mod.rs:150-168`), the Rust reference `perception.wgsl:289-311`'s
/// `legacy_encoding` mirrors and `gpu/fly_map/fly_map.wgsl`'s `eye_pixel_drive` copies a
/// third time (WGSL cannot `use` across shader modules — same reason
/// `retina_digest.wgsl:148-149` carries its own literal copy). Dark (Road, very negative)
/// clamps to `brightness = 0` and drives at full `gain`; bright (a near Wall, +127) drives
/// at 0 — "dark drives strongly, because the fly's photoreceptors inhibit downstream cells
/// in the light" (task spec), the sign a real photoreceptor has and a distance-encoded
/// pixel does not carry on its own; reusing `legacy_encoding` as the light proxy is a
/// deliberate choice documented here, not a measured fact about fly vision.
pub fn eye_drive(gain: f32, frame: &EyeFrame, being: usize, neuron: usize) -> Option<f32> {
    let pixel = *frame.pixel_of_neuron.get(neuron)?;
    if pixel == NONE {
        return None;
    }
    let packed = *frame
        .retina
        .get(being * frame.retina_stride + pixel as usize)?;
    let dist_m = (packed >> 8) as f32 / 1000.0;
    let material = (packed & 0xFF) as u8;
    let raw = crate::gpu::legacy_encoding(crate::gpu::Pixel { dist_m, material });
    // The encoded pixel is a full i8: road is around -128, grass +40, a near wall +127. Mapping
    // only [0, 127] onto brightness would clamp EVERY road pixel to 0 whatever its distance
    // (measured 14/09: road spans raw -128..-104 from 2 m to 300 m), and the track is most of
    // what the retina sees — the eye would be blind to the road. Map the whole i8 range instead.
    let brightness = ((raw as f32 + 128.0) / 255.0).clamp(0.0, 1.0);
    Some(gain * (1.0 - brightness))
}

impl FlyMap {
    /// Whether this map carries the pixel-binding block — what `--fly-eye` requires
    /// (`docs/FLY-MAP-BRAIN.md` "Eye binding"). A v1 file, or a v2 file exported without
    /// the pixel-binding pass on the Python side, answers `false`.
    pub fn has_pixel_binding(&self) -> bool {
        self.pixel_of_neuron.is_some()
    }

    /// A six-neuron map like [`FlyMap::synthetic`], with three neurons additionally bound
    /// to known retina pixels: neuron 0 → pixel 5, neuron 3 → pixel 900, neuron 4 → the
    /// LAST pixel of the real `RETINA_TOTAL = 1764` stride (`RETINA_TOTAL - 1` = 1763) —
    /// so an off-by-one in the stride a test wires up shows as an out-of-bounds read on
    /// the very neuron this map exists to exercise. Neurons 1, 2 and 5 are left at
    /// [`NONE`]: not eye-driven, so a test can confirm the genome's `in_bind` still
    /// governs them exactly as it does today.
    pub fn synthetic_with_eye() -> FlyMap {
        let mut map = FlyMap::synthetic();
        let mut pixel_of_neuron = vec![NONE; map.n()];
        pixel_of_neuron[0] = 5;
        pixel_of_neuron[3] = 900;
        pixel_of_neuron[4] = crate::config::iara::vision::RETINA_TOTAL as u32 - 1;
        map.pixel_of_neuron = Some(pixel_of_neuron);
        map
    }

    /// Whether this map carries the reflex-readout block — what `--fly-readout reflex`
    /// requires (`docs/FLY-MAP-BRAIN.md` "Reflex readout"). A v1/v2 file, or a v3 file
    /// exported without the reflex-readout pass, answers `false`.
    pub fn has_reflex_readout(&self) -> bool {
        self.reflex_out.is_some()
    }

    /// A six-neuron map like [`FlyMap::synthetic`], additionally carrying a
    /// reflex-readout block: throttle `(NONE, NONE)` — the map does not produce it, the
    /// host supplies `--fly-cruise-throttle` — steer `(plus = group 0, minus = group 1)`,
    /// brake `(plus = group 1, minus = group 0)`. Steer and brake are deliberately
    /// swapped rather than identical, so a test that mixed the two command slots up
    /// fails instead of passing by symmetry.
    pub fn synthetic_with_reflex() -> FlyMap {
        let mut map = FlyMap::synthetic();
        map.reflex_out = Some([NONE, NONE, 0, 1, 1, 0]);
        map
    }
}

/// Whether the reflex-readout block's throttle pair is `(NONE, NONE)` — "the map does
/// not produce this command; the host supplies it" (`docs/FLY-MAP-BRAIN.md` "Reflex
/// readout"). Pure, so the substitution `apply_reflex_host_override` makes is
/// unit-testable without a running `gpu_evolve` process; `context.rs` calls it once per
/// tick (cheap: two comparisons) rather than caching the answer, since the map never
/// changes after load and the cost of asking again is noise next to a tick.
pub fn reflex_throttle_is_none(reflex_out: [u32; OUT_SLOTS]) -> bool {
    reflex_out[0] == NONE && reflex_out[1] == NONE
}

/// The host-side half of `--fly-readout reflex` (`docs/FLY-MAP-BRAIN.md` "Reflex
/// readout"), applied to one being's raw `[throttle, steer, brake]` command AFTER
/// `fly_read`/`cpu_read` computed it from the (possibly overridden) `out_pair`:
///
/// * throttle is replaced by `cruise` exactly when `throttle_is_none` (from
///   [`reflex_throttle_is_none`]) says the map produces no throttle command — otherwise
///   the map's own command passes through unchanged;
/// * steer is multiplied by `sign` (`--fly-reflex-sign`, {1, -1}) — the sign convention
///   that maps "motion on the left" to a left or right turn is not documented anywhere
///   in this repository, so this is an experiment arm, not a guess;
/// * brake always passes through unchanged.
///
/// Pure, so `context.rs`'s FLY BRAIN OVERRIDE site is a thin per-being wrapper a test
/// does not have to open a GPU to exercise. With `throttle_is_none = false` and
/// `sign = 1.0` (the defaults when `--fly-readout reflex` is off, or its throttle slot
/// is bound) this returns `cmd` unchanged: `x * 1.0` is bit-exact for every finite `x`,
/// so a run that never asks for the reflex arm sees the historical numbers.
pub fn apply_reflex_host_override(
    cmd: [f32; crate::fly_map::COMMANDS],
    throttle_is_none: bool,
    cruise: f32,
    sign: f32,
) -> [f32; crate::fly_map::COMMANDS] {
    [
        if throttle_is_none { cruise } else { cmd[0] },
        cmd[1] * sign,
        cmd[2],
    ]
}

/// Whether a run started with `--fly-readout reflex` can proceed against `map`
/// (`docs/FLY-MAP-BRAIN.md` "Reflex readout"): no map, or a map without the block, both
/// refuse — the same two-step shape `--fly-eye` already checks against
/// `has_pixel_binding`. Pure and returning `Err(String)` rather than printing and
/// exiting itself, so the refusal is unit-testable; `main()` prints the message and
/// exits non-zero on `Err`.
pub fn readout_reflex_requires_the_block(map: Option<&FlyMap>) -> Result<(), String> {
    match map {
        None => Err(
            "--fly-readout reflex without --fly-map: there is no map to read the reflex \
             block from."
                .to_string(),
        ),
        Some(m) if !m.has_reflex_readout() => Err(
            "--fly-readout reflex requires a flymap v3 file with the reflex-readout \
             block (flags bit 3); the loaded map carries none."
                .to_string(),
        ),
        Some(_) => Ok(()),
    }
}

/// `flags` this map's `write()` must set — the mirror of [`read_trailing_blocks`].
pub(super) fn compute_flags(map: &FlyMap) -> u32 {
    let mut flags = 0u32;
    if map.pre_csr.is_some() {
        flags |= FLAG_PRESYNAPTIC_CSR;
    }
    if map.pixel_of_neuron.is_some() {
        flags |= FLAG_PIXEL_BINDING;
    }
    if map.no_refractory.is_some() {
        flags |= FLAG_NO_REFRACTORY;
    }
    if map.reflex_out.is_some() {
        flags |= FLAG_REFLEX_READOUT;
    }
    flags
}

/// Reads the trailing blocks named by `flags`, in the fixed order of §1: presynaptic CSR,
/// then pixel binding, then the no-refractory bitmap, then the reflex-readout block.
/// `flags == 0` (every v1 file, and a v2/v3 file exported with nothing extra) reads
/// nothing and returns four `None`s.
///
/// Each read names its own block on a truncated file, rather than surfacing the bare
/// "failed to fill whole buffer" `io::Error` the v1 fields already have to live with — the
/// blocks this function owns are new, so they do not inherit that pre-existing gap.
pub(super) fn read_trailing_blocks(
    f: &mut impl Read,
    path: &str,
    n: usize,
    m: usize,
    flags: u32,
) -> std::io::Result<(
    Option<PresynapticCsr>,
    Option<Vec<u32>>,
    Option<Vec<u32>>,
    Option<[u32; OUT_SLOTS]>,
)> {
    let pre_csr = if flags & FLAG_PRESYNAPTIC_CSR != 0 {
        let indptr = named_u32s(f, n + 1, path, "presynaptic CSR indptr")?;
        let indices = named_u32s(f, m, path, "presynaptic CSR indices")?;
        let weights = named_f32s(f, m, path, "presynaptic CSR weights")?;
        Some(PresynapticCsr {
            indptr,
            indices,
            weights,
        })
    } else {
        None
    };
    let pixel_of_neuron = if flags & FLAG_PIXEL_BINDING != 0 {
        Some(named_u32s(f, n, path, "pixel_of_neuron")?)
    } else {
        None
    };
    let no_refractory = if flags & FLAG_NO_REFRACTORY != 0 {
        Some(named_u32s(f, n.div_ceil(32), path, "no_refractory bitmap")?)
    } else {
        None
    };
    let reflex_out = if flags & FLAG_REFLEX_READOUT != 0 {
        let raw = named_u32s(f, OUT_SLOTS, path, "reflex_out")?;
        // `named_u32s(f, OUT_SLOTS, ..)` reads exactly `OUT_SLOTS` words or fails, so this
        // conversion cannot panic.
        Some(raw.try_into().expect("named_u32s returned OUT_SLOTS words"))
    } else {
        None
    };
    Ok((pre_csr, pixel_of_neuron, no_refractory, reflex_out))
}

/// Writes the trailing blocks present on `map`, in the same fixed order
/// [`read_trailing_blocks`] reads them. A no-op when all four are `None` (a v1 file).
pub(super) fn write_trailing_blocks(f: &mut impl Write, map: &FlyMap) -> std::io::Result<()> {
    if let Some(csr) = &map.pre_csr {
        write_u32s(f, &csr.indptr)?;
        write_u32s(f, &csr.indices)?;
        for w in &csr.weights {
            f.write_all(&w.to_le_bytes())?;
        }
    }
    if let Some(pix) = &map.pixel_of_neuron {
        write_u32s(f, pix)?;
    }
    if let Some(bitmap) = &map.no_refractory {
        write_u32s(f, bitmap)?;
    }
    if let Some(reflex_out) = &map.reflex_out {
        write_u32s(f, reflex_out)?;
    }
    Ok(())
}

fn named_u32s(f: &mut impl Read, n: usize, path: &str, what: &str) -> std::io::Result<Vec<u32>> {
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf).map_err(|e| {
        bad_data(format!(
            "{path}: truncated flymap v2 file — could not read the {what} block \
             ({n} x u32, {e})"
        ))
    })?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn named_f32s(f: &mut impl Read, n: usize, path: &str, what: &str) -> std::io::Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf).map_err(|e| {
        bad_data(format!(
            "{path}: truncated flymap v2 file — could not read the {what} block \
             ({n} x f32, {e})"
        ))
    })?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
#[path = "fly_map_eye_tests.rs"]
mod fly_map_eye_tests;

#[cfg(test)]
#[path = "fly_map_reflex_tests.rs"]
mod fly_map_reflex_tests;
