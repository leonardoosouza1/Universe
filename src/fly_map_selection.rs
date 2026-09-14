//! The genome side of the fly map: what a being SELECTS on top of the fixed connectome.
//!
//! Split out of `fly_map.rs` to keep both files under the 600-line cap of
//! `docs/STYLE.md` §2, along the seam a reader already perceives: `fly_map.rs` is the MAP
//! (its file format and its dynamics), this file is the SELECTION (the shape of the
//! genome block, the operators that move it, and the per-being tables the kernels read).
//! Everything here is re-exported by `fly_map`, so callers see one module.

use std::sync::OnceLock;

use super::*;

/// Shape of the loaded map, as far as the genome is concerned.
///
/// The `fly` block cannot size itself: its length is `IN_CHANNELS + 8 + n_types`, and
/// `n_types` comes from the map header. `Genome::random` therefore reads this process-wide
/// value, which `gpu_evolve` sets once, right after loading the map and before the first
/// genome is born. Unset (no `--fly-map`) means every `fly` block is empty and nothing
/// downstream allocates — the flag-off path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlyLayout {
    /// Afferent groups a perception channel may bind to.
    pub k_in: usize,
    /// Efferent groups a command may be read from.
    pub k_out: usize,
    /// Cell types, i.e. the length of the pruning mask.
    pub n_types: usize,
}

impl FlyLayout {
    /// Length of the genome's `fly` block under this layout.
    pub fn block_len(&self) -> usize {
        OFF_TYPE_MASK + self.n_types
    }
}

static LAYOUT: OnceLock<FlyLayout> = OnceLock::new();

/// Publishes the loaded map's shape. Idempotent by `OnceLock`: the second call is
/// ignored, which is correct because one process loads one map.
pub fn set_layout(l: FlyLayout) {
    let _ = LAYOUT.set(l);
}

/// The shape of the loaded map, or `None` when no map was loaded in this process.
pub fn layout() -> Option<FlyLayout> {
    LAYOUT.get().copied()
}

/// The per-being tables the kernels read, already flattened (§3, "per generation the host
/// packs").
///
/// Being-minor everywhere, like the state buffers: the row of being `b` starts at
/// `b * stride`. Built once per generation by [`pack_selection`].
#[derive(Clone, Debug, Default)]
pub struct FlySelection {
    /// `n_beings × k_in` — which perception channel drives each afferent group, or [`NONE`].
    pub group_to_input: Vec<u32>,
    /// `n_beings × 6` — (plus, minus) efferent group per command.
    pub out_pair: Vec<u32>,
    /// `n_beings × 2` — input gain and synaptic scale.
    pub gain_scale: Vec<f32>,
    /// `n_beings × n_types` — 1 = type on, 0 = pruned.
    pub type_mask: Vec<f32>,
    /// Beings covered by the tables above.
    pub n_beings: usize,
    /// Layout the tables were packed against.
    pub layout: FlyLayout,
}

impl FlySelection {
    /// Input gain of being `b`.
    pub fn gain(&self, b: usize) -> f32 {
        self.gain_scale[b * 2]
    }
    /// Synaptic scale of being `b`.
    pub fn scale(&self, b: usize) -> f32 {
        self.gain_scale[b * 2 + 1]
    }
}

/// Turns one `fly` genome block per being into the tables of §3.
///
/// The inversion `in_bind[channel] -> group` into `group_to_input[group] -> channel` is
/// where a collision can happen: two channels may bind the same afferent group and only
/// one can drive it. **The higher channel wins**, by writing later in the ascending scan —
/// the rule the spec fixes, written here once so neither kernel has to know about it.
///
/// A block of the wrong length (an old checkpoint, or a map swapped between runs) packs as
/// fully unbound with mask 1 and gains 1: inert, not garbage.
pub fn pack_selection(blocks: &[&[f32]], layout: FlyLayout, n_beings: usize) -> FlySelection {
    let mut sel = FlySelection {
        group_to_input: vec![NONE; n_beings * layout.k_in],
        out_pair: vec![NONE; n_beings * OUT_SLOTS],
        gain_scale: vec![1.0; n_beings * 2],
        type_mask: vec![1.0; n_beings * layout.n_types],
        n_beings,
        layout,
    };
    for b in 0..n_beings {
        let block = match blocks.get(b) {
            Some(v) if v.len() == layout.block_len() => *v,
            _ => continue,
        };
        for ch in 0..IN_CHANNELS {
            let g = block[ch];
            if g < 0.0 {
                continue;
            }
            let g = g as usize;
            if g < layout.k_in {
                // Ascending scan: a later (higher) channel overwrites an earlier one.
                sel.group_to_input[b * layout.k_in + g] = ch as u32;
            }
        }
        for s in 0..OUT_SLOTS {
            let g = block[OFF_OUT_PAIR + s];
            sel.out_pair[b * OUT_SLOTS + s] = if g >= 0.0 && (g as usize) < layout.k_out {
                g as u32
            } else {
                NONE
            };
        }
        sel.gain_scale[b * 2] = block[OFF_GAIN];
        sel.gain_scale[b * 2 + 1] = block[OFF_SCALE];
        for t in 0..layout.n_types {
            // Anything that is not "on" prunes: the gene is a switch, and a mutation that
            // left 0.5 in it must not become half a neuron.
            sel.type_mask[b * layout.n_types + t] = if block[OFF_TYPE_MASK + t] >= 0.5 {
                1.0
            } else {
                0.0
            };
        }
    }
    sel
}

/// Keeps the fly draws off the sequence that produced a being's weights — the same
/// precaution `random_sensors_within_ranges` takes for the sensor bank. Written once, so
/// every birth site (`Genome::random`, and `gpu_evolve` conforming a resumed population)
/// gives the same being the same selection.
const BIRTH_SEED_OFFSET: u64 = 0xF1A0_0000_F1A0_0000;

/// The `fly` block a newborn genome carries: a random selection when a map is loaded,
/// nothing at all when there is none.
pub fn birth_block(seed: u64) -> Vec<f32> {
    match layout() {
        Some(l) => random_block(seed.wrapping_add(BIRTH_SEED_OFFSET), l),
        None => Vec::new(),
    }
}

/// A newborn `fly` block (§3): each channel bound to a random group with probability
/// [`BIRTH_BIND_P`], a random efferent group in each of the six slots, gains at 1 and the
/// whole mask on.
///
/// PURE (takes the seed, reads no atomic) for the same reason as
/// `evolution::random_sensors_within_ranges`: a test must not have to toggle a global.
pub fn random_block(seed: u64, layout: FlyLayout) -> Vec<f32> {
    let mut rng = crate::rng::Lcg::new(seed);
    let mut block = vec![0.0f32; layout.block_len()];
    for ch in 0..IN_CHANNELS {
        block[ch] = if layout.k_in > 0 && rng.bernoulli(BIRTH_BIND_P) {
            draw_group(&mut rng, layout.k_in) as f32
        } else {
            -1.0
        };
    }
    for s in 0..OUT_SLOTS {
        block[OFF_OUT_PAIR + s] = if layout.k_out > 0 {
            draw_group(&mut rng, layout.k_out) as f32
        } else {
            -1.0
        };
    }
    block[OFF_GAIN] = 1.0;
    block[OFF_SCALE] = 1.0;
    for t in 0..layout.n_types {
        block[OFF_TYPE_MASK + t] = 1.0;
    }
    block
}

/// Mutates a `fly` block in place (§3). Called only under `evolution::FLY_MUTATE_ON`.
///
/// Four operators, one per slice, because the four slices are four different kinds of
/// gene: a binding is a discrete choice (redraw, sometimes to unbound), the gains are
/// scale parameters (log-normal, so ×1.1 means the same thing anywhere), and the mask is a
/// pruning operator with an asymmetric rate — switching a type off is cheaper to try than
/// switching it back on, which is the shape `neuromod-lab/docs/94` found selection can
/// actually exploit.
///
/// PURE, like [`random_block`]: the generator comes from outside.
pub fn mutate_block(
    block: &mut [f32],
    layout: FlyLayout,
    p_bind: f32,
    p_off: f32,
    p_on: f32,
    sigma: f32,
    rng: &mut impl crate::gpu::evolution::GenoRng,
) {
    if block.len() != layout.block_len() {
        return;
    }
    for ch in 0..IN_CHANNELS {
        if rng.chance() < p_bind {
            block[ch] = if layout.k_in == 0 || rng.chance() < REBIND_TO_NONE_P {
                -1.0
            } else {
                floor_index(rng.chance(), layout.k_in)
            };
        }
    }
    for s in 0..OUT_SLOTS {
        if rng.chance() < p_bind && layout.k_out > 0 {
            block[OFF_OUT_PAIR + s] = floor_index(rng.chance(), layout.k_out);
        }
    }
    // Log-normal: a gain cannot cross zero and flip the sign of every input at once.
    block[OFF_GAIN] = (block[OFF_GAIN] * (rng.gauss() * sigma).exp()).clamp(0.0, 10.0);
    block[OFF_SCALE] = (block[OFF_SCALE] * (rng.gauss() * sigma).exp()).clamp(0.0, 10.0);
    for t in 0..layout.n_types {
        let gene = &mut block[OFF_TYPE_MASK + t];
        if *gene >= 0.5 {
            if rng.chance() < p_off {
                *gene = 0.0;
            }
        } else if rng.chance() < p_on {
            *gene = 1.0;
        }
    }
}

/// Density of the pruning mask: the fraction of cell types still switched on.
pub fn mask_density(block: &[f32], layout: FlyLayout) -> f32 {
    if block.len() != layout.block_len() || layout.n_types == 0 {
        return 0.0;
    }
    let on = block[OFF_TYPE_MASK..OFF_TYPE_MASK + layout.n_types]
        .iter()
        .filter(|&&v| v >= 0.5)
        .count();
    on as f32 / layout.n_types as f32
}

/// The channels this block binds, as `(channel, group)` pairs, in channel order.
pub fn bound_channels(block: &[f32], layout: FlyLayout) -> Vec<(usize, u32)> {
    if block.len() != layout.block_len() {
        return Vec::new();
    }
    (0..IN_CHANNELS)
        .filter(|&ch| block[ch] >= 0.0)
        .map(|ch| (ch, block[ch] as u32))
        .collect()
}

/// `u ∈ [0, 1)` to a group index in `0..k`, as an f32 gene. `min` guards the u = 1.0 that
/// a float rounding can still produce.
fn floor_index(u: f32, k: usize) -> f32 {
    (u * k as f32).floor().min((k - 1) as f32)
}

fn draw_group(rng: &mut crate::rng::Lcg, k: usize) -> usize {
    floor_index(rng.uniform(), k) as usize
}
