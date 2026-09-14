// ── FLY MAP: the Drosophila connectome as a fixed brain ───────────────────────
//
// Two kernels, one module (docs/FLY-MAP-BRAIN.md §4):
//
//   fly_step — one WAVE per NEURON, its 32 lanes being 32 BEINGS. The map is shared by
//              every being, so the wave that owns CSR row j reads that row once and serves
//              the whole population with it. That is the whole reason the state is
//              being-minor (`x[j * B + b]`): the row is read once, and the 32 lanes then
//              read 32 CONSECUTIVE floats. See "Why lanes are beings" below.
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

// ── Why lanes are beings (the mapping, and the numbers that forced it) ────────
//
// The first version of `fly_step` ran ONE THREAD PER NEURON with the beings in an inner
// loop over a private `acc[64]` array. Measured on the RX 6750 XT (RDNA2, wave32), per
// tick: full map (n = 138 639, m = 15.1 M) 20.1 ms at B = 8 and 98.1 ms at B = 40;
// `--min-weight 3` (4.9 M edges) 11.6 / 57.1 ms; `_w3_nooptic` (2.15 M edges) 5.1 ms at
// B = 8. Cost was LINEAR in B — about 2.4 ms per extra being on the full map — which is
// the signature of a memory path that gets no wider as the population grows. Two causes:
//
//   1. NO COALESCING. With one thread per neuron, the 32 threads of a wave own 32
//      DIFFERENT rows, so at any instant they are reading 32 unrelated `indices[e]`, and
//      `r_old[indices[e] * B + b]` lands in 32 unrelated cache lines. Every being in the
//      inner loop paid for its own scattered gather: 32 transactions per edge per being.
//   2. DIVERGENCE. Those 32 rows have 32 different lengths (the connectome is heavy
//      tailed), so the whole wave ran at the speed of its longest row, 32 rows at a time.
//      And `acc[64]` indexed by a loop variable is the textbook case the compiler spills
//      to scratch memory.
//
// The mapping below inverts it. `@workgroup_size(32, 4)`: the 32 lanes of x are 32 BEINGS
// and the 4 rows of y are 4 consecutive NEURONS, so one wave (32 invocations on RDNA2,
// linearised as x-major) is exactly one neuron serving 32 beings. Consequences:
//
//   * `indices[e]` and `weights[e]` are UNIFORM across the wave — one broadcast fetch of
//     the row, not 32 divergent ones. Edge traffic stays at m x 8 B = 120 MB per tick
//     whatever B is, because the two accumulators below let one pass over the row serve
//     64 beings.
//   * `r_old[indices[e] * B + b]` with `b = lane` means lane k reads word k of a
//     contiguous run — the 32 loads fold into one or two 128-byte transactions instead of
//     32. The whole `r_old` is n x B x 4 = 22 MB at B = 40, so it lives in the 96 MB
//     Infinity Cache and the gather is served there rather than from VRAM.
//   * Zero divergence inside a wave: all 32 lanes walk the SAME row, so the loop trip
//     count is uniform and nobody waits for a neighbour's longer row.
//   * `acc` is two scalars in registers (`acc_a`, `acc_b`), never an indexed array, so
//     there is nothing to spill.
//
// At 432 GB/s the 120 MB of edges is ~0.28 ms and the gather is cache-resident, which is
// why the arithmetic said the full map at B = 40 should cost single-digit milliseconds
// and not 98. The per-tick figures this mapping actually reached are in the commit that
// introduced it and in `bench_test.rs`, which is the way to re-measure.
//
// Semantics are UNCHANGED and bit-for-bit: each being still accumulates its row in
// ascending edge order e0..e1, so the sum is the same sequence of f32 multiply-adds the
// CPU reference performs (`universe::fly_map::cpu_step`). Only WHO does it moved.
//
// That is MEASURED, not argued. Both kernels ran five chained ticks at B = 40 on the full
// map and `debug_read_state` was dumped: 5 545 560 floats, 5 458 454 of them non-zero,
// BYTE-IDENTICAL (md5 389e1ac0e612783395c73ae24c8c5fa4 both sides). The six-neuron
// synthetic map in `parity_test` could not have shown this on its own — its rows are too
// short for a reordered sum to round differently — which is why the real map was used.

// Mirrors `universe::fly_map::NONE` (`src/fly_map.rs`), the only other place this value is
// written. The host also ships it in `params.none_id`, which is what the code below reads —
// this constant is the documentation of the contract, not a second source of truth.
const NONE_ID: u32 = 0xFFFFFFFFu;

// Lanes per wave on RDNA2, and therefore beings per chunk. Changing this breaks the
// coalescing argument above; it is not a tuning knob, it is the hardware's wave width.
const LANES: u32 = 32u;

// Consecutive neurons per workgroup, i.e. waves per workgroup. 4 x 32 = 128 invocations,
// the same workgroup size the kernel had before, which keeps occupancy where it was.
const ROWS: u32 = 4u;

// Beings served by ONE pass over the CSR row: two chunks of `LANES`, one per accumulator
// register. A population larger than this repeats the pass, at the cost of re-reading the
// row — the same trade the old `MAX_B` made, at 64 for the same reason.
const BEINGS_PER_PASS: u32 = 64u;

// Workgroups a single dispatch dimension may carry (`maxComputeWorkgroupsPerDimension`).
// The host splits the row-groups across x and y when they do not fit in x alone; at
// n = 138 639 they do (34 660 groups), so y is 1 and this is insurance, not a code path
// that runs today.
const MAX_GROUPS_PER_DIM: u32 = 65535u;

struct FlyParams {
    n: u32,               // neurons
    n_beings: u32,        // beings alive in this dispatch
    k_in: u32,            // afferent groups
    n_types: u32,         // cell types (pruning mask width)
    alpha: f32,           // 1 / tau_steps
    wave_stride_u32: u32, // u32 words per being in the sensory wave (48 = 192 channels)
    none_id: u32,         // NONE_ID, shipped by the host
    k_out: u32,           // efferent groups
    eye_on: u32,          // `--fly-eye` gate (docs/FLY-MAP-BRAIN.md "Eye binding"); 0/1
    retina_stride: u32,   // pixels per being in `retina` — RETINA_TOTAL, host-injected
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
// ── Eye binding (`--fly-eye`, docs/FLY-MAP-BRAIN.md "Eye binding") — bound only into
// `fly_step`'s layout, the same way `wave` (binding 6) is: `fly_read` never touches them.
@group(0) @binding(17) var<storage, read>       pixel_of_neuron: array<u32>;
@group(0) @binding(18) var<storage, read>       retina:          array<u32>;

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

// Eye binding (`--fly-eye`, docs/FLY-MAP-BRAIN.md "Eye binding"): what the rendered
// retina drives into a neuron bound to `pixel`, replacing the genome's channel binding
// for THAT neuron. `u_eye = gain * (1 - brightness)`, `brightness = clamp((pixel_value +
// 128) / 255, 0, 1)` — dark (Road, very negative) clamps to brightness 0 and drives at
// full gain; bright (a near Wall, +127) drives at 0.
//
// `pixel_value` is decoded and encoded exactly as the 192 sensory channels are, three
// literal copies of the same formula because WGSL cannot `use` across shader modules
// (the same reason `retina_digest.wgsl:148-149` carries its own copy of
// `legacy_encoding`):
//   * `retina` packs `(mm << 8) | material` per pixel — `src/gpu/retina.wgsl:159-160`.
//   * the decode (`pixel_dist_m` / `pixel_mat`) — `src/gpu/perception.wgsl:270-277`.
//   * `legacy_encoding(d, mat)` — `src/gpu/perception.wgsl:289-311`, mirrored on the CPU
//     by `universe::gpu::legacy_encoding` (`src/gpu/mod.rs:150-168`), the function
//     `fly_map::eye_drive` (`src/fly_map_eye.rs`) calls directly — THAT copy and this one
//     are what `fly_map_parity` (`gpu/fly_map/parity_test.rs`) checks agree.
//
// `retina` is BEING-MAJOR (`retina[being * retina_stride + pixel]`), unlike `r_old`/`x`
// (neuron-major, being-minor): the gather below is therefore NOT the coalesced access
// the rest of this kernel is built around (`being` varies per lane, so consecutive lanes
// land `retina_stride` words apart) — one transaction per lane instead of one for the
// wave. This is a deliberate, UNMEASURED trade-off: the map's own layout is fixed by
// `retina.wgsl`, eye-driven neurons are expected to be a small minority of `n`, and the
// branch that reaches this function is wave-uniform (every lane's `pixel_of_neuron[j]`
// is the same value, so either the whole wave takes this path or none of it does — no
// thread divergence, only the scattered gather itself).
fn eye_pixel_drive(being: u32, pixel: u32) -> f32 {
    let packed: u32 = retina[being * params.retina_stride + pixel];
    let mm: u32 = packed >> 8u;
    let mat: u32 = packed & 0xFFu;
    let d: f32 = f32(mm) / 1000.0;
    let depth_fade: f32 = clamp(d / 1200.0, 0.0, 1.0);
    var v: f32;
    if mat == 2u { // Wall
        if d < 30.0 {
            v = 127.0 - (sqrt(clamp(d / 30.0, 0.0, 1.0)) * 72.0);
        } else if d < 120.0 {
            v = 55.0 - (clamp((d - 30.0) / 90.0, 0.0, 1.0) * 55.0);
        } else if d < 1200.0 {
            v = 0.0 - (clamp((d - 120.0) / 480.0, 0.0, 1.0) * 25.0);
        } else {
            v = -25.0 - (clamp((d - 1200.0) / 1200.0, 0.0, 1.0) * 25.0);
        }
    } else if mat == 0u { // Grass
        v = 40.0 - (depth_fade * 40.0);
    } else if mat == 1u { // Road
        v = -128.0 + (depth_fade * 96.0);
    } else { // Hole (3)
        v = 90.0 - (clamp(d / 50.0, 0.0, 1.0) * 30.0);
    }
    // `trunc` because the CPU reference ends in `v.clamp(-128.0, 127.0) as i8`, and a Rust float
    // cast truncates toward zero. Without it the two implementations disagree by up to one unit
    // of the encoding — invisible while the negative half was being clamped to zero, and caught
    // by `fly_map_parity_with_the_eye_on` the moment the full range started being used (14/09).
    let raw: f32 = trunc(clamp(v, -128.0, 127.0));
    // Full i8 range, not [0, 127] — see the comment in `fly_map_eye.rs::eye_drive`: clamping the
    // negative half would make every road pixel identical and the eye blind to the track.
    let brightness: f32 = clamp((raw + 128.0) / 255.0, 0.0, 1.0);
    return gain_scale[being * 2u] * (1.0 - brightness);
}

// The §2 integration for ONE (neuron, being): leaky integrate the gathered drive, then
// emit through the pruning mask. Split out only so the two chunks below do not repeat it;
// the arithmetic is identical to `cpu_step`'s, in the same order.
//
// `row` is `j * n_beings`, hoisted by the caller because it does not depend on the being.
// `j` is the neuron index itself, needed here (not derivable from `row`/`b` alone without
// a division) only to look up `pixel_of_neuron[j]` — the eye binding is per NEURON, not
// per afferent group, so it cannot be folded into `g` the way `input_drive` is.
fn integrate(row: u32, j: u32, b: u32, g: u32, t: u32, acc: f32) {
    let idx: u32 = row + b;
    let xo: f32 = x[idx];
    // Eye binding replaces the genome's channel drive for THIS neuron when both the
    // `--fly-eye` gate is on (`params.eye_on`) and the map bound this neuron to a pixel
    // (`pixel != NONE_ID`) — `params.eye_on == 0u` (the default) takes the exact same
    // `input_drive` branch this kernel had before the eye extension existed, so the
    // arithmetic below is byte-for-byte what it was with the flag off.
    let pixel: u32 = pixel_of_neuron[j];
    var u: f32;
    if params.eye_on != 0u && pixel != params.none_id {
        u = eye_pixel_drive(b, pixel);
    } else {
        u = input_drive(b, g);
    }
    let xn: f32 = xo + params.alpha * (-xo + gain_scale[b * 2u + 1u] * acc + u);
    x[idx] = xn;
    // A pruned cell type emits nothing. The state keeps integrating behind the mask —
    // pruning silences the output, it does not kill the neuron.
    r_new[idx] = tanh(xn) * type_mask[b * params.n_types + t];
}

@compute @workgroup_size(32, 4)
fn fly_step(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) n_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // One wave = one neuron. `lid.y` picks the wave inside the workgroup (invocations are
    // linearised x-major, and x is exactly one wave wide), `lid.x` the being inside it.
    let group: u32 = wg.y * n_wg.x + wg.x;
    let j: u32 = group * ROWS + lid.y;
    if j >= params.n {
        return;
    }
    let lane: u32 = lid.x;
    let bn: u32 = params.n_beings;
    let e0: u32 = indptr[j];
    let e1: u32 = indptr[j + 1u];
    let g: u32 = in_group_of[j];
    let t: u32 = type_of[j];
    let row: u32 = j * bn;
    let last: u32 = bn - 1u; // `bn >= 1`: the host returns early on an empty population

    var b0: u32 = 0u;
    loop {
        if b0 >= bn {
            break;
        }
        let b_a: u32 = b0 + lane;
        let b_b: u32 = b0 + LANES + lane;
        // A lane past the end of the population still loads, clamped to the last being,
        // and throws the result away at the write below. Clamping instead of branching
        // keeps the edge loop free of control flow, and clamping to the NEIGHBOUR rather
        // than to 0 keeps the dead lanes inside the same cache line as the live ones.
        let i_a: u32 = min(b_a, last);
        let i_b: u32 = min(b_b, last);
        var acc_a: f32 = 0.0;
        var acc_b: f32 = 0.0;

        // The gather: one CSR row, read once, serving up to 64 beings. `indices[e] * bn`
        // is the being-minor base of the presynaptic neuron's PREVIOUS output, and the
        // lane index that follows is what makes the 32 loads consecutive.
        //
        // `b0 + LANES < bn` is uniform over the whole dispatch, so this is a branch the
        // wave takes together — the second chunk is not merely predicated off, it is not
        // fetched at all when the population does not reach it.
        if b0 + LANES < bn {
            for (var e: u32 = e0; e < e1; e = e + 1u) {
                let w: f32 = weights[e];
                let base: u32 = indices[e] * bn;
                acc_a = acc_a + w * r_old[base + i_a];
                acc_b = acc_b + w * r_old[base + i_b];
            }
        } else {
            for (var e: u32 = e0; e < e1; e = e + 1u) {
                acc_a = acc_a + weights[e] * r_old[indices[e] * bn + i_a];
            }
        }

        if b_a < bn {
            integrate(row, j, b_a, g, t, acc_a);
        }
        if b_b < bn {
            integrate(row, j, b_b, g, t, acc_b);
        }
        b0 = b0 + BEINGS_PER_PASS;
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
