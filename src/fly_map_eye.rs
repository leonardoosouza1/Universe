//! The `flymap` v2 trailing blocks, and the "Eye binding" extension of §2 that reads one
//! of them.
//!
//! Split out of `fly_map.rs` for the 600-line cap (`docs/STYLE.md` §2), along the seam a
//! reader already perceives: `fly_map.rs` is v1 (the map and the rate model as they were
//! before this change), this file is everything v2 added. Re-exported by `fly_map`, so
//! callers see one module.
//!
//! Two of the three v2 blocks — the presynaptic CSR and the no-refractory bitmap — are
//! parsed and stored here but read by NOTHING in this file: `docs/FLY-MAP-BRAIN.md` §1 is
//! explicit that they exist for a future spiking kernel, not the rate path this task
//! implements. Only the pixel-binding block feeds the dynamics, through [`eye_drive`].

use std::io::{Read, Write};

use super::*;

/// Bit 0 of the v2 header's `flags` word: the presynaptic CSR trails the v1 body.
pub const FLAG_PRESYNAPTIC_CSR: u32 = 1 << 0;
/// Bit 1: the pixel-binding block (`pixel_of_neuron`) trails it.
pub const FLAG_PIXEL_BINDING: u32 = 1 << 1;
/// Bit 2: the no-refractory bitmap trails it.
pub const FLAG_NO_REFRACTORY: u32 = 1 << 2;

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
    let brightness = (raw as f32 / 127.0).clamp(0.0, 1.0);
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
    flags
}

/// Reads the v2 trailing blocks named by `flags`, in the fixed order of §1: presynaptic
/// CSR, then pixel binding, then the no-refractory bitmap. `flags == 0` (every v1 file, and
/// a v2 file exported with nothing extra) reads nothing and returns three `None`s.
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
) -> std::io::Result<(Option<PresynapticCsr>, Option<Vec<u32>>, Option<Vec<u32>>)> {
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
    Ok((pre_csr, pixel_of_neuron, no_refractory))
}

/// Writes the v2 trailing blocks present on `map`, in the same fixed order
/// [`read_trailing_blocks`] reads them. A no-op when all three are `None` (a v1 file).
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
