// ── FLY MAP: the Drosophila connectome as a fixed brain ───────────────────────
//
// Two kernels, one module (docs/FLY-MAP-BRAIN.md §4):
//
//   fly_step — one thread per NEURON, looping over beings inside. The map is shared by
//              every being, so the thread that owns CSR row j reads that row once and
//              serves the whole population with it. That is the whole reason the state is
//              being-minor (`x[j * B + b]`): the row is read once, the states are read
//              contiguously.
//   fly_read — one thread per (being, command): the mean over the plus efferent group
//              minus the mean over the minus group.
//
// The CPU reference for both is `universe::fly_map::{cpu_step, cpu_read}`, and
// `gpu::fly_map::parity_test` compares them on a synthetic map. Any change here that is
// not made there too is a divergence the parity test exists to catch.
//
// Buffers are read-only except `x`, `r_new` and `cmd`. `r` is double-buffered on the host
// (two bind groups, swapped every tick), because a neuron reads its neighbours' PREVIOUS
// output: writing into the same buffer would make the result depend on the order in which
// the GPU happened to schedule the rows.

// Mirrors `universe::fly_map::NONE` (`src/fly_map.rs`), the only other place this value is
// written. The host also ships it in `params.none_id`, which is what the code below reads —
// this constant is the documentation of the contract, not a second source of truth.
const NONE_ID: u32 = 0xFFFFFFFFu;

// Beings served per pass of the inner loop. The private `acc` array has to be a
// compile-time size; a population larger than this is handled by repeating the loop, at
// the cost of re-reading the CSR row once per chunk.
const MAX_B: u32 = 64u;

struct FlyParams {
    n: u32,               // neurons
    n_beings: u32,        // beings alive in this dispatch
    k_in: u32,            // afferent groups
    n_types: u32,         // cell types (pruning mask width)
    alpha: f32,           // 1 / tau_steps
    wave_stride_u32: u32, // u32 words per being in the sensory wave (48 = 192 channels)
    none_id: u32,         // NONE_ID, shipped by the host
    k_out: u32,           // efferent groups
};

@group(0) @binding(0)  var<uniform>             params:         FlyParams;
@group(0) @binding(1)  var<storage, read>       indptr:         array<u32>;
@group(0) @binding(2)  var<storage, read>       indices:        array<u32>;
@group(0) @binding(3)  var<storage, read>       weights:        array<f32>;
@group(0) @binding(4)  var<storage, read>       type_of:        array<u32>;
@group(0) @binding(5)  var<storage, read>       in_group_of:    array<u32>;
@group(0) @binding(6)  var<storage, read>       wave:           array<u32>;
@group(0) @binding(7)  var<storage, read>       group_to_input: array<u32>;
@group(0) @binding(8)  var<storage, read>       gain_scale:     array<f32>;
@group(0) @binding(9)  var<storage, read>       type_mask:      array<f32>;
@group(0) @binding(10) var<storage, read>       r_old:          array<f32>;
@group(0) @binding(11) var<storage, read_write> r_new:          array<f32>;
@group(0) @binding(12) var<storage, read_write> x:              array<f32>;
@group(0) @binding(13) var<storage, read>       out_indptr:     array<u32>;
@group(0) @binding(14) var<storage, read>       out_members:    array<u32>;
@group(0) @binding(15) var<storage, read>       out_pair:       array<u32>;
@group(0) @binding(16) var<storage, read_write> cmd:            array<f32>;

// One sensory channel of one being, sign-extended from the packed wave.
//
// The wave arrives 4 i8 channels per u32, exactly as perception writes it and as
// `slow_gather.wgsl` already reads it; the stride comes from the host so this kernel does
// not become the seventh hand-written copy of WAVE_U32S (docs/STYLE.md §4).
fn wave_i8(being: u32, ch: u32) -> f32 {
    let word: u32 = wave[being * params.wave_stride_u32 + ch / 4u];
    let byte: u32 = (word >> ((ch % 4u) * 8u)) & 0xFFu;
    if byte >= 128u {
        return f32(i32(byte) - 256);
    }
    return f32(i32(byte));
}

// The `u` term of §2: what the bound perception channel drives into a neuron of afferent
// group `g`, or 0 when the group is NONE or nothing is bound to it.
fn input_drive(being: u32, g: u32) -> f32 {
    if g == params.none_id || g >= params.k_in {
        return 0.0;
    }
    let ch: u32 = group_to_input[being * params.k_in + g];
    if ch == params.none_id {
        return 0.0;
    }
    return gain_scale[being * 2u] * wave_i8(being, ch) / 127.0;
}

@compute @workgroup_size(128)
fn fly_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j: u32 = gid.x;
    if j >= params.n {
        return;
    }
    let e0: u32 = indptr[j];
    let e1: u32 = indptr[j + 1u];
    let g: u32 = in_group_of[j];
    let t: u32 = type_of[j];
    let scale_off: u32 = 1u;

    var acc: array<f32, 64>;
    var b0: u32 = 0u;
    loop {
        if b0 >= params.n_beings {
            break;
        }
        let hi: u32 = min(b0 + MAX_B, params.n_beings);
        for (var b: u32 = b0; b < hi; b = b + 1u) {
            acc[b - b0] = 0.0;
        }
        // The gather: one CSR row, every being. `indices[e] * n_beings + b` is the
        // being-minor address of the presynaptic neuron's PREVIOUS output.
        for (var e: u32 = e0; e < e1; e = e + 1u) {
            let w: f32 = weights[e];
            let base: u32 = indices[e] * params.n_beings;
            for (var b: u32 = b0; b < hi; b = b + 1u) {
                acc[b - b0] = acc[b - b0] + w * r_old[base + b];
            }
        }
        for (var b: u32 = b0; b < hi; b = b + 1u) {
            let idx: u32 = j * params.n_beings + b;
            let xo: f32 = x[idx];
            let xn: f32 = xo
                + params.alpha
                    * (-xo + gain_scale[b * 2u + scale_off] * acc[b - b0] + input_drive(b, g));
            x[idx] = xn;
            // A pruned cell type emits nothing. The state keeps integrating behind the
            // mask — pruning silences the output, it does not kill the neuron.
            r_new[idx] = tanh(xn) * type_mask[b * params.n_types + t];
        }
        b0 = b0 + MAX_B;
    }
}

// Mean activation of an efferent group for one being. An unbound, unknown or empty group
// is exactly 0, which is what makes an unevolved command slot inert instead of random.
fn group_mean(g: u32, being: u32) -> f32 {
    if g == params.none_id || g >= params.k_out {
        return 0.0;
    }
    let s: u32 = out_indptr[g];
    let e: u32 = out_indptr[g + 1u];
    if e <= s {
        return 0.0;
    }
    var sum: f32 = 0.0;
    for (var i: u32 = s; i < e; i = i + 1u) {
        sum = sum + r_new[out_members[i] * params.n_beings + being];
    }
    return sum / f32(e - s);
}

@compute @workgroup_size(64)
fn fly_read(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot: u32 = gid.x;
    let being: u32 = slot / 3u;
    let c: u32 = slot % 3u;
    if being >= params.n_beings {
        return;
    }
    let plus: u32 = out_pair[being * 6u + c * 2u];
    let minus: u32 = out_pair[being * 6u + c * 2u + 1u];
    cmd[being * 3u + c] = group_mean(plus, being) - group_mean(minus, being);
}
