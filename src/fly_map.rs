//! The Drosophila connectome as a fixed brain: map file, rate model, CPU reference.
//!
//! Why this module exists: the map is never generated and never trained here. It is
//! loaded once, shared by every being, and evolution only SELECTS on top of it — which
//! perception channel drives which afferent cell type, which efferent cell types are read
//! as throttle/steer/brake, which cell types stay switched on, and two scalar gains
//! (`docs/FLY-MAP-BRAIN.md`). Everything in this file is GPU-free on purpose: the
//! numerical contract of the kernels in `gpu::fly_map` is defined here, in Rust, so a
//! parity test can compare them without anything else being in doubt.
//!
//! Two things live in this file and nowhere else:
//!
//! 1. the `flymap` v1 binary format ([`FlyMap::load`]), including the writer the tests use
//!    to build a synthetic map instead of reading the 120 MB real one;
//! 2. the rate model of §2 ([`cpu_step`], [`cpu_read`]) — the reference the WGSL mirrors.
//!
//! The third piece, the genome block of §3 ([`random_block`], [`mutate_block`],
//! [`pack_selection`]), is in `fly_map_selection.rs` and re-exported here: one module for
//! the caller, two files for the 600-line cap. It is not in `gpu::evolution` because that
//! file is already well over that cap.
//!
//! A fourth piece, the v2 trailing blocks and the eye binding they enable (`--fly-eye`,
//! `docs/FLY-MAP-BRAIN.md` "Eye binding"), is in `fly_map_eye.rs` for the same reason and
//! is likewise re-exported here.
//!
//! The sentinel [`NONE`] is defined here once and mirrored in exactly one other place,
//! `gpu/fly_map/fly_map.wgsl`, with a comment pointing back at this constant.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::iara::brain::SENSORY_WAVE_SIZE;

// The genome block and the per-being tables (§3). A separate file, re-exported here so
// callers see one module — the 600-line cap of `docs/STYLE.md` §2 is what splits them,
// and the seam is map (this file) against selection (that one).
#[path = "fly_map_selection.rs"]
mod selection;
pub use selection::{
    FlyLayout, FlySelection, apply_reflex_readout, birth_block, bound_channels, layout,
    mask_density, mutate_block, pack_selection, random_block, set_layout,
};

// The v2 trailing blocks (presynaptic CSR, pixel binding, no-refractory bitmap) and the
// eye-drive extension of §2 that reads the pixel-binding one. Split out for the same
// 600-line reason as `selection`; re-exported so callers see one module.
#[path = "fly_map_eye.rs"]
mod eye;
pub use eye::{
    EyeFrame, FLAG_NO_REFRACTORY, FLAG_PIXEL_BINDING, FLAG_PRESYNAPTIC_CSR, FLAG_REFLEX_READOUT,
    PresynapticCsr, apply_reflex_host_override, eye_drive, readout_reflex_requires_the_block,
    reflex_throttle_is_none,
};

/// "no binding" / "no group": the u32 sentinel of the map file and of both kernels.
///
/// Mirrored once, in `gpu/fly_map/fly_map.wgsl` (`NONE_ID`), because WGSL cannot import a
/// Rust constant; the host also ships it in the params uniform, so the shader literal is a
/// fallback and not a second source of truth. `pixel_of_neuron` (v2) reuses the same
/// sentinel for "not driven by the eye".
pub const NONE: u32 = 0xFFFF_FFFF;

/// Magic of a `flymap` v1 file — no trailing blocks, `flags` (v2 sense) always 0.
pub const MAGIC_V1: &[u8; 8] = b"FLYMAP01";
/// Magic of a `flymap` v2 file — the v1 body plus the optional trailing blocks named by
/// `flags` (docs/FLY-MAP-BRAIN.md §1). [`FlyMap::load`] accepts both; a v1 file loads with
/// every v2 field `None`.
pub const MAGIC_V2: &[u8; 8] = b"FLYMAP02";

/// Gate for the eye binding of `--fly-eye` (docs/FLY-MAP-BRAIN.md "Eye binding"). Default
/// `false`: with the flag off, `cpu_step`'s caller passes `eye: None` and the kernel's
/// `params.eye_on` is 0, so neither reads the pixel-binding buffers at all — the same
/// "last consumer decides nothing when off" shape as `gpu::evolution::FLY_MUTATE_ON`.
///
/// Read by `main()` (which also requires the loaded map to carry the pixel-binding block
/// before setting this) and by `context.rs::tick()` (which decides the `eye_on` argument
/// of `FlyMapGpu::dispatch`); `cpu_step` itself takes `eye: Option<&EyeFrame>` explicitly
/// rather than reading this atomic, so the numerical reference stays a pure function like
/// `random_block`/`mutate_block` — only the ORCHESTRATION layer (test or binary) reads the
/// gate to decide what to pass in.
pub static FLY_EYE_ON: AtomicBool = AtomicBool::new(false);

/// Whether `--fly-eye` is engaged in this process.
pub fn eye_on() -> bool {
    FLY_EYE_ON.load(Ordering::Relaxed)
}

/// Gate for `--fly-readout reflex` (docs/FLY-MAP-BRAIN.md "Reflex readout"). Default
/// `false` (`genome`, today's behaviour); `true` (`reflex`) makes `apply_reflex_readout`
/// overwrite `out_pair` for every being with the map's own `reflex_out` block, once per
/// generation — same "off is never consulted" shape as [`FLY_EYE_ON`].
pub static FLY_READOUT_REFLEX_ON: AtomicBool = AtomicBool::new(false);

/// Whether `--fly-readout reflex` is engaged in this process.
pub fn readout_reflex_on() -> bool {
    FLY_READOUT_REFLEX_ON.load(Ordering::Relaxed)
}

/// Perception channels a genome can bind to an afferent group — the full sensory wave.
///
/// Reads `SENSORY_WAVE_SIZE` (`config.rs`) rather than the literal 192 of the spec: the
/// wave already shrank once from 223 and two hand-written copies were missed
/// (`docs/STYLE.md` §4).
pub const IN_CHANNELS: usize = SENSORY_WAVE_SIZE;

/// Commands read out of the map every tick: throttle, steer, brake.
pub const COMMANDS: usize = 3;

/// Efferent group slots per genome: a (plus, minus) pair per command.
pub const OUT_SLOTS: usize = COMMANDS * 2;

/// Offset of `out_pair[6]` inside the genome's `fly` block.
pub const OFF_OUT_PAIR: usize = IN_CHANNELS;
/// Offset of the input `gain` gene.
pub const OFF_GAIN: usize = OFF_OUT_PAIR + OUT_SLOTS;
/// Offset of the synaptic `scale` gene.
pub const OFF_SCALE: usize = OFF_GAIN + 1;
/// Offset of `type_mask[n_types]`.
pub const OFF_TYPE_MASK: usize = OFF_SCALE + 1;

/// Probability that a channel is born bound to a random afferent group (§3).
pub const BIRTH_BIND_P: f32 = 0.25;
/// Probability that a rebinding draw unbinds the channel instead (§3).
pub const REBIND_TO_NONE_P: f32 = 0.2;

/// A fixed sparse map, CSR by POSTSYNAPTIC neuron.
///
/// `indptr[j]..indptr[j+1]` are the input edges of neuron `j`; `indices[e]` is the
/// presynaptic neuron and `weights[e]` the signed synapse count (+ excitatory,
/// − inhibitory), exactly as FlyWire/Shiu et al. 2024 publish them.
#[derive(Clone, Debug, Default)]
pub struct FlyMap {
    /// Cell-type id per neuron, `0..n_types`.
    pub type_of: Vec<u32>,
    /// Afferent group per neuron, or [`NONE`].
    pub in_group_of: Vec<u32>,
    /// CSR row offsets, `n + 1` entries.
    pub indptr: Vec<u32>,
    /// Presynaptic neuron per edge.
    pub indices: Vec<u32>,
    /// Signed synapse count per edge.
    pub weights: Vec<f32>,
    /// Efferent group membership offsets, `k_out + 1` entries.
    pub out_indptr: Vec<u32>,
    /// Neurons of each efferent group, indexed by `out_indptr`.
    pub out_members: Vec<u32>,
    /// Number of cell types (length of the pruning mask).
    pub n_types: usize,
    /// Number of afferent groups.
    pub k_in: usize,

    // ── v2 trailing blocks (docs/FLY-MAP-BRAIN.md §1) — all optional, all absent from a
    // v1 file. Stored on `FlyMap` so one load produces one value the whole run reads from;
    // none of them are re-derived from the postsynaptic CSR.
    /// Presynaptic CSR (flag bit 0). Stored, never read by the rate path — a future
    /// spiking kernel needs it.
    pub pre_csr: Option<PresynapticCsr>,
    /// The rendered-retina pixel driving each neuron, or [`NONE`] (flag bit 1). Read only
    /// by the eye extension of `cpu_step` and by `FlyMapGpu::new`'s upload.
    pub pixel_of_neuron: Option<Vec<u32>>,
    /// Packed bitmap, `ceil(n / 32)` words, one bit per neuron (flag bit 2). Stored, never
    /// read by the rate path — a future spiking kernel needs it.
    pub no_refractory: Option<Vec<u32>>,
    /// (plus, minus) efferent GROUP per command — throttle, steer, brake, or [`NONE`]
    /// meaning "the map does not produce this command" (flag bit 3, `docs/FLY-MAP-
    /// BRAIN.md` "Reflex readout"). Read at the HOST level, by `apply_reflex_readout`.
    pub reflex_out: Option<[u32; OUT_SLOTS]>,
}

impl FlyMap {
    /// Neurons.
    pub fn n(&self) -> usize {
        self.type_of.len()
    }
    /// Edges.
    pub fn m(&self) -> usize {
        self.indices.len()
    }
    /// Efferent groups.
    pub fn k_out(&self) -> usize {
        self.out_indptr.len().saturating_sub(1)
    }
    /// Shape the genome block needs.
    pub fn fly_layout(&self) -> FlyLayout {
        FlyLayout {
            k_in: self.k_in,
            k_out: self.k_out(),
            n_types: self.n_types,
        }
    }
    /// Bytes the map itself occupies on the GPU (the per-being state is not counted).
    ///
    /// Counts `pixel_of_neuron` when present — it becomes a real `FlyMapGpu` buffer
    /// (`gpu/fly_map/mod.rs::buf_pixel_of_neuron`). Does NOT count `pre_csr` or
    /// `no_refractory`: nothing uploads them (§1, "Store... but do not use them").
    pub fn resident_bytes(&self) -> usize {
        (self.indptr.len()
            + self.indices.len()
            + self.weights.len()
            + self.type_of.len()
            + self.in_group_of.len()
            + self.out_indptr.len()
            + self.out_members.len()
            + self.pixel_of_neuron.as_ref().map_or(0, Vec::len))
            * 4
    }

    /// Reads a `flymap` v1 or v2 file (§1). A v1 file (magic `FLYMAP01`) loads through
    /// this same function with every v2 field `None` — v2 only ADDS optional trailing
    /// blocks after the v1 body, it never changes how that body is read.
    ///
    /// Fails with a message naming the export that writes the format rather than a bare
    /// parse error: the caller is at a terminal with a run that will not start.
    pub fn load(path: &str) -> std::io::Result<FlyMap> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        let is_v1 = &magic == MAGIC_V1;
        if !is_v1 && &magic != MAGIC_V2 {
            return Err(bad_data(format!(
                "{path}: not a flymap file (magic {magic:?} is neither FLYMAP01 nor \
                 FLYMAP02). Produce it with neuromod-lab/src/fly_export.py, which writes \
                 the header this loader expects."
            )));
        }
        let head = read_u32s(&mut f, 8)?;
        let (n, m, n_types, k_in, k_out, out_len, flags) = (
            head[0] as usize,
            head[1] as usize,
            head[2] as usize,
            head[3] as usize,
            head[4] as usize,
            head[5] as usize,
            head[6],
        );
        if is_v1 && flags != 0 {
            return Err(bad_data(format!(
                "{path}: magic is FLYMAP01 (v1) but header word 6 (flags, v2 sense) is \
                 {flags:#x} instead of 0 — a v1 file never sets a v2 flag bit. Truncated or \
                 mislabeled export."
            )));
        }
        if n == 0 || n_types == 0 {
            return Err(bad_data(format!(
                "{path}: header says n={n} neurons and n_types={n_types}; both must be > 0."
            )));
        }
        let indptr = read_u32s(&mut f, n + 1)?;
        let indices = read_u32s(&mut f, m)?;
        let weights = read_f32s(&mut f, m)?;
        let type_of = read_u32s(&mut f, n)?;
        let in_group_of = read_u32s(&mut f, n)?;
        let out_indptr = read_u32s(&mut f, k_out + 1)?;
        let out_members = read_u32s(&mut f, out_len)?;
        let (pre_csr, pixel_of_neuron, no_refractory, reflex_out) =
            eye::read_trailing_blocks(&mut f, path, n, m, flags)?;
        let map = FlyMap {
            type_of,
            in_group_of,
            indptr,
            indices,
            weights,
            out_indptr,
            out_members,
            n_types,
            k_in,
            pre_csr,
            pixel_of_neuron,
            no_refractory,
            reflex_out,
        };
        map.validate(path)?;
        Ok(map)
    }

    /// Structural checks, run once at load.
    ///
    /// They cost one pass over `m` next to a load that already read `m` floats, and they
    /// are the only thing standing between a truncated export and a kernel indexing past
    /// the end of a storage buffer — which WGSL clamps silently, so the symptom would be
    /// wrong numbers and no error at all.
    fn validate(&self, path: &str) -> std::io::Result<()> {
        let n = self.n();
        if self.indptr.len() != n + 1 {
            return Err(bad_data(format!(
                "{path}: indptr has {} entries, expected n + 1 = {}",
                self.indptr.len(),
                n + 1
            )));
        }
        if self.indptr.last().copied().unwrap_or(0) as usize != self.m() {
            return Err(bad_data(format!(
                "{path}: indptr[n] = {} but the file carries {} edges — truncated export",
                self.indptr.last().copied().unwrap_or(0),
                self.m()
            )));
        }
        if self.indptr.windows(2).any(|w| w[1] < w[0]) {
            return Err(bad_data(format!("{path}: indptr is not monotonic")));
        }
        if self.indices.iter().any(|&i| i as usize >= n) {
            return Err(bad_data(format!(
                "{path}: an edge names a presynaptic neuron outside 0..{n}"
            )));
        }
        if self.type_of.iter().any(|&t| t as usize >= self.n_types) {
            return Err(bad_data(format!(
                "{path}: a neuron carries a cell type outside 0..{}",
                self.n_types
            )));
        }
        if self
            .in_group_of
            .iter()
            .any(|&g| g != NONE && g as usize >= self.k_in)
        {
            return Err(bad_data(format!(
                "{path}: a neuron carries an afferent group outside 0..{} and is not NONE",
                self.k_in
            )));
        }
        if self.out_indptr.last().copied().unwrap_or(0) as usize != self.out_members.len() {
            return Err(bad_data(format!(
                "{path}: out_indptr[k_out] = {} does not match out_members_len = {}",
                self.out_indptr.last().copied().unwrap_or(0),
                self.out_members.len()
            )));
        }
        if self.out_members.iter().any(|&i| i as usize >= n) {
            return Err(bad_data(format!(
                "{path}: an efferent group names a neuron outside 0..{n}"
            )));
        }
        if let Some(csr) = &self.pre_csr {
            let m = self.m();
            if csr.indptr.len() != n + 1 {
                return Err(bad_data(format!(
                    "{path}: presynaptic CSR indptr has {} entries, expected n + 1 = {}",
                    csr.indptr.len(),
                    n + 1
                )));
            }
            // §1: `pre_indices u32[m]`, `pre_weights f32[m]` — the SAME m as the
            // postsynaptic CSR, because a presynaptic CSR re-indexes the same edges by
            // their OTHER endpoint, it does not carry a different edge set.
            if csr.indices.len() != m || csr.weights.len() != m {
                return Err(bad_data(format!(
                    "{path}: presynaptic CSR has {} indices and {} weights, expected m = {m} \
                     for both — it must re-index the SAME edges as the postsynaptic CSR",
                    csr.indices.len(),
                    csr.weights.len()
                )));
            }
            if csr.indptr.last().copied().unwrap_or(0) as usize != m {
                return Err(bad_data(format!(
                    "{path}: presynaptic CSR indptr[n] = {} but the file carries m = {m} \
                     edges — truncated export",
                    csr.indptr.last().copied().unwrap_or(0)
                )));
            }
            if csr.indices.iter().any(|&i| i as usize >= n) {
                return Err(bad_data(format!(
                    "{path}: the presynaptic CSR names a postsynaptic neuron outside 0..{n}"
                )));
            }
        }
        if let Some(pix) = &self.pixel_of_neuron {
            if pix.len() != n {
                return Err(bad_data(format!(
                    "{path}: pixel_of_neuron has {} entries, expected n = {n}",
                    pix.len()
                )));
            }
        }
        if let Some(bitmap) = &self.no_refractory {
            let expect = n.div_ceil(32);
            if bitmap.len() != expect {
                return Err(bad_data(format!(
                    "{path}: no_refractory has {} words, expected ceil(n / 32) = {expect}",
                    bitmap.len()
                )));
            }
        }
        let k_out = self.k_out();
        if let Some(r) = &self.reflex_out {
            if r.iter().any(|&g| g != NONE && g as usize >= k_out) {
                return Err(bad_data(format!(
                    "{path}: reflex_out names an efferent group outside 0..{k_out}"
                )));
            }
        }
        Ok(())
    }

    /// Writes this map — the round trip [`FlyMap::load`] reads. Magic and `flags` follow
    /// from what is populated: with `pre_csr`, `pixel_of_neuron` and `no_refractory` all
    /// `None` the output is byte-for-byte the historical v1 file (magic `FLYMAP01`,
    /// `flags` word 0); any of the three present writes v2 (`FLYMAP02`) with the matching
    /// bit set and that block appended, in the fixed order of §1.
    ///
    /// Exists for the tests: they build a small map instead of depending on the 120 MB
    /// real export, and a writer that shares no parsing code with the reader is what makes
    /// a format test prove something.
    pub fn write(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let flags = eye::compute_flags(self);
        f.write_all(if flags == 0 { MAGIC_V1 } else { MAGIC_V2 })?;
        let head: [u32; 8] = [
            self.n() as u32,
            self.m() as u32,
            self.n_types as u32,
            self.k_in as u32,
            self.k_out() as u32,
            self.out_members.len() as u32,
            flags,
            0,
        ];
        for v in head {
            f.write_all(&v.to_le_bytes())?;
        }
        write_u32s(&mut f, &self.indptr)?;
        write_u32s(&mut f, &self.indices)?;
        for w in &self.weights {
            f.write_all(&w.to_le_bytes())?;
        }
        write_u32s(&mut f, &self.type_of)?;
        write_u32s(&mut f, &self.in_group_of)?;
        write_u32s(&mut f, &self.out_indptr)?;
        write_u32s(&mut f, &self.out_members)?;
        eye::write_trailing_blocks(&mut f, self)?;
        f.flush()
    }

    /// A six-neuron map: signed weights, three cell types (so a pruned type is a visible
    /// hole), one bound and one unbound afferent group, two efferent groups. Small enough
    /// to compute by hand and wide enough to exercise every branch of §2.
    pub fn synthetic() -> FlyMap {
        // Rows are POSTSYNAPTIC. Mixed signs, so a sign error cannot cancel itself out.
        let rows: [&[(u32, f32)]; 6] = [
            &[(5, 0.5)],
            &[(0, -1.0)],
            &[(0, 0.75), (1, 0.25)],
            &[(2, -0.5)],
            &[(2, 1.0), (3, 0.5)],
            &[(4, -0.25), (1, 0.3)],
        ];
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for row in rows {
            for &(src, w) in row {
                indices.push(src);
                weights.push(w);
            }
            indptr.push(indices.len() as u32);
        }
        FlyMap {
            type_of: vec![0, 0, 1, 1, 2, 2],
            in_group_of: vec![0, 0, 1, NONE, NONE, NONE],
            indptr,
            indices,
            weights,
            out_indptr: vec![0, 2, 4],
            out_members: vec![4, 5, 2, 3],
            n_types: 3,
            k_in: 2,
            pre_csr: None,
            pixel_of_neuron: None,
            no_refractory: None,
            reflex_out: None,
        }
    }
}

/// One step of the rate model (§2), for every being at once.
///
/// `x`, `r_old` and `r_new` are `n × n_beings`, being-minor (`x[j * B + b]`), so the thread
/// that owns row `j` reads the CSR row once and serves every being — the layout the kernel
/// needs, kept identical here so the parity test compares the same arithmetic and not two
/// different orders of summation.
///
/// `wave` is the raw sensory wave, `n_beings × IN_CHANNELS` i8 values as perception writes
/// them; the division by 127 is the same one the kernel does.
///
/// `eye` is the "Eye binding" extension (docs/FLY-MAP-BRAIN.md): `None` reproduces the
/// pre-`--fly-eye` formula exactly (the branch below is simply never taken, so the
/// generated float sequence is identical to the version of this function before the
/// extension existed). `Some(frame)` makes every neuron with
/// `frame.pixel_of_neuron[j] != NONE` read its `u` term from the retina instead of from
/// `wave` via the genome's binding — see [`eye_drive`].
pub fn cpu_step(
    map: &FlyMap,
    sel: &FlySelection,
    wave: &[i8],
    alpha: f32,
    x: &mut [f32],
    r_old: &[f32],
    r_new: &mut [f32],
    eye: Option<&EyeFrame>,
) {
    let b_n = sel.n_beings;
    let k_in = sel.layout.k_in;
    let n_types = sel.layout.n_types;
    for j in 0..map.n() {
        let e0 = map.indptr[j] as usize;
        let e1 = map.indptr[j + 1] as usize;
        let g = map.in_group_of[j];
        let t = map.type_of[j] as usize;
        for b in 0..b_n {
            let mut acc = 0.0f32;
            for e in e0..e1 {
                acc += map.weights[e] * r_old[map.indices[e] as usize * b_n + b];
            }
            let u = match eye.and_then(|frame| eye_drive(sel.gain(b), frame, b, j)) {
                Some(v) => v,
                None => input_drive(sel, wave, b, g, k_in),
            };
            let idx = j * b_n + b;
            let xn = x[idx] + alpha * (-x[idx] + sel.scale(b) * acc + u);
            x[idx] = xn;
            r_new[idx] = xn.tanh() * sel.type_mask[b * n_types + t];
        }
    }
}

/// The `u` term of §2: what the bound perception channel drives into this neuron, or 0.
fn input_drive(sel: &FlySelection, wave: &[i8], b: usize, group: u32, k_in: usize) -> f32 {
    if group == NONE || group as usize >= k_in {
        return 0.0;
    }
    let ch = sel.group_to_input[b * k_in + group as usize];
    if ch == NONE {
        return 0.0;
    }
    sel.gain(b) * wave[b * IN_CHANNELS + ch as usize] as f32 / 127.0
}

/// The three commands per being (§2): mean over the plus group minus mean over the minus
/// group. An unbound, unknown or empty group contributes exactly 0.
pub fn cpu_read(map: &FlyMap, sel: &FlySelection, r: &[f32]) -> Vec<[f32; COMMANDS]> {
    let b_n = sel.n_beings;
    let mut out = vec![[0.0f32; COMMANDS]; b_n];
    for b in 0..b_n {
        for c in 0..COMMANDS {
            let plus = group_mean(map, r, b_n, b, sel.out_pair[b * OUT_SLOTS + c * 2]);
            let minus = group_mean(map, r, b_n, b, sel.out_pair[b * OUT_SLOTS + c * 2 + 1]);
            out[b][c] = plus - minus;
        }
    }
    out
}

fn group_mean(map: &FlyMap, r: &[f32], b_n: usize, b: usize, g: u32) -> f32 {
    if g == NONE || g as usize >= map.k_out() {
        return 0.0;
    }
    let s = map.out_indptr[g as usize] as usize;
    let e = map.out_indptr[g as usize + 1] as usize;
    if e <= s {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &i in &map.out_members[s..e] {
        sum += r[i as usize * b_n + b];
    }
    sum / (e - s) as f32
}

fn bad_data(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

fn read_u32s(f: &mut impl Read, n: usize) -> std::io::Result<Vec<u32>> {
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn read_f32s(f: &mut impl Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn write_u32s(f: &mut impl Write, v: &[u32]) -> std::io::Result<()> {
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "fly_map_tests.rs"]
mod fly_map_tests;
