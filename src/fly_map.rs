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
//! The sentinel [`NONE`] is defined here once and mirrored in exactly one other place,
//! `gpu/fly_map/fly_map.wgsl`, with a comment pointing back at this constant.

use std::io::{Read, Write};

use crate::config::iara::brain::SENSORY_WAVE_SIZE;

// The genome block and the per-being tables (§3). A separate file, re-exported here so
// callers see one module — the 600-line cap of `docs/STYLE.md` §2 is what splits them,
// and the seam is map (this file) against selection (that one).
#[path = "fly_map_selection.rs"]
mod selection;
pub use selection::{
    FlyLayout, FlySelection, birth_block, bound_channels, layout, mask_density, mutate_block,
    pack_selection, random_block, set_layout,
};

/// "no binding" / "no group": the u32 sentinel of the map file and of both kernels.
///
/// Mirrored once, in `gpu/fly_map/fly_map.wgsl` (`NONE_ID`), because WGSL cannot import a
/// Rust constant; the host also ships it in the params uniform, so the shader literal is a
/// fallback and not a second source of truth.
pub const NONE: u32 = 0xFFFF_FFFF;

/// Magic of the `flymap` v1 file, written by `neuromod-lab/src/fly_export.py`.
pub const MAGIC: &[u8; 8] = b"FLYMAP01";

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
    pub fn resident_bytes(&self) -> usize {
        (self.indptr.len()
            + self.indices.len()
            + self.weights.len()
            + self.type_of.len()
            + self.in_group_of.len()
            + self.out_indptr.len()
            + self.out_members.len())
            * 4
    }

    /// Reads a `flymap` v1 file (§1).
    ///
    /// Fails with a message naming the export that writes the format rather than a bare
    /// parse error: the caller is at a terminal with a run that will not start.
    pub fn load(path: &str) -> std::io::Result<FlyMap> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad_data(format!(
                "{path}: not a flymap v1 file (magic {magic:?} is not FLYMAP01). Produce it \
                 with neuromod-lab/src/fly_export.py, which writes the header this loader \
                 expects."
            )));
        }
        let head = read_u32s(&mut f, 8)?;
        let (n, m, n_types, k_in, k_out, out_len) = (
            head[0] as usize,
            head[1] as usize,
            head[2] as usize,
            head[3] as usize,
            head[4] as usize,
            head[5] as usize,
        );
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
        Ok(())
    }

    /// Writes this map in the `flymap` v1 format — the round trip [`FlyMap::load`] reads.
    ///
    /// Exists for the tests: they build a six-neuron map instead of depending on the
    /// 120 MB real export, and a writer that shares no parsing code with the reader is
    /// what makes a format test prove something.
    pub fn write(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        let head: [u32; 8] = [
            self.n() as u32,
            self.m() as u32,
            self.n_types as u32,
            self.k_in as u32,
            self.k_out() as u32,
            self.out_members.len() as u32,
            0,
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
pub fn cpu_step(
    map: &FlyMap,
    sel: &FlySelection,
    wave: &[i8],
    alpha: f32,
    x: &mut [f32],
    r_old: &[f32],
    r_new: &mut [f32],
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
            let u = input_drive(sel, wave, b, g, k_in);
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
