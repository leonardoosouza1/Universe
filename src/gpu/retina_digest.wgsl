// RETINA DIGEST — perception CfC + min_wall_ahead + edge_row6/7, all on the GPU.
//
// # Why this kernel exists
//
// `context.rs::tick()` (gpu_evolve) used to do, EVERY tick, a 340 KB readback (48 beings ×
// 1764 px × 4 B) of the entire retina and process 84,672 pixels in a Rust loop just to
// produce, per being: 16 floats of CfC state (ch192-207), `min_wall_ahead` (life drain
// from walls) and `edge_row6`/`edge_row7` (ch99/ch100). That readback+loop was the
// measured bottleneck of the tick (2.30 ms versus 0.63 ms for the pure GPU tick).
//
// This kernel runs AFTER `retina.wgsl`, in the SAME `CommandEncoder`, consuming the
// physical retina buffer already in VRAM — with NO round-trip to the CPU in the hot
// loop. Only the small result goes down to the CPU: 3 floats per being (`extra_out`).
//
// # What this kernel does NOT do
//
// The life loop (wall_ticks, life drain, stagnation, fuel, -1 tick of
// life_left) STAYS on the CPU — it merely starts consuming `min_wall_ahead` from here
// instead of recomputing it from the pixels. See `context.rs::tick()`, the block titled
// "Atualiza CfC percepção" ("update perception CfC" — the block name is still in the // check-language: allow
// original Portuguese there), now reduced to: read `extra_out`, apply the drains.
//
// # The CfC equation — ONE source, two places that mirror it
//
// The equation itself lives in `iara_cfc::CfcCell::step` (external crate, shared by the
// whole IARA ecosystem) — see `src/math/cfc_perception.rs`, which is this crate's front
// door to it and documents the formula in full. THIS shader is the GPU mirror; if the
// equation changes, BOTH copies (there and here) must change together — there is no
// third copy anywhere (Leonardo: "no scattered CfC").
//
// matvec here uses a sequential `acc` (`for c { acc += w[r,c]*x[c] }`), replicating the
// loop of `iara_cfc::matvec` — do NOT use WGSL's `dot()`, which allows the compiler to
// reassociate the sum and diverge in ULPs from the result computed on the CPU.
//
// # Pipeline until it becomes x[48] (CfC input) — mirrors `perception/retina.rs`
//
// 1. `legacy_encoding(dist_m, material)` per pixel — LITERAL COPY of
//    `perception.wgsl:251` (WGSL does not import functions across files in this project;
//    any fix must touch both copies, plus `gpu::legacy_encoding` in
//    Rust — the same debt documented in `perception.wgsl` for a 4th copy).
// 2. Col max-pool: 126 columns → 16 bins per row (`cols_per_bin = 126/16 = 7`, the
//    last bin absorbs the remainder up to 126) — mirrors `RetinaSensor::sense_retina`.
//    The max is taken over the SIGNED value (i32, same range as an i8), and only AFTER
//    shifted to [0,255] via `+128` (equivalent to `(max_val as i16 + 128) as u8`).
// 3. Zone-pool: far=rows[0,5) MEAN, mid=rows[5,10) MEAN, near=rows[10,14) MAX —
//    mirrors `retina_zones()`. Each pooled value (0..255) becomes f32 via `/127.5 - 1.0`
//    DURING the zone aggregation (not before) — mirrors `ZoneConfig::pool`.
// 4. x = [far(16) | mid(16) | near(16)] = 48 floats — same order as `pool_flat()`.
//
// # extra_out[3..19] ("ch192-207") — the SAME 1-tick lag that already existed
//
// The old CPU injected `hidden[]` into `buf_temporal[33-48]` BEFORE the dispatch (CfC
// state computed in the PREVIOUS tick, via post-submit readback) and only `perception_temporal
// .wgsl` converted that hidden into a channel (writing to `output`, ch192-207 — see finding
// P1.8 below on why this is NO LONGER reproduced here). This kernel preserves only
// the LAG (not the value's destination):
//   1) computes the 16 values using `cfc_state` AS IT WAS at the start of the tick (= h after
//      tick t-1), writes them to `extra_out[3..19]`;
//   2) only AFTER THAT processes this tick's retina and writes the NEW h back to
//      `cfc_state`, for the NEXT tick to read.
// The hidden→channel conversion formula is the x/(1+|x|) approximation that was already in
// `perception_temporal.wgsl` (NOT WGSL's native `tanh()`) — copied literally
// below so as not to change the computed value, only the destination buffer (see finding P1.8).
//
// # Bindings
//
// binding 0 (uniform)          params: n_beings
// binding 1 (storage, read)    retina: array<u32>       — N × 1764, packed (mm<<8|mat)
// binding 2 (storage, read)    cfc_weights: array<f32>   — 1312 floats, a SINGLE buffer
//   shared by all beings (fixed, identical weights — see the finding "CfC weights
//   never loaded from the genome" in src/math/cfc_perception.rs). Layout:
//   [w_xi(768) | w_hi(256) | w_f(256) | b_i(16) | b_f(16)].
// binding 3 (storage, read_write) cfc_state: array<f32>  — N × 16, RESIDENT in VRAM
//   between ticks. Zeroed by `reset()` (host) between generations.
// binding 4 (storage, read_write) output: array<atomic<u32>> — the SAME `buf_output` of
//   perception (perception.rs). THIS KERNEL NO LONGER WRITES HERE (see finding P1.8
//   below) — only read implicitly by other kernels in the same pipeline; kept
//   in the bind group because the layout is shared with `perception.rs`.
// binding 5 (storage, read_write) extra_out: array<f32>  — N × EXTRA_STRIDE(19):
//   [0] min_wall_ahead (m, large sentinel if there is no wall ahead in the central
//       columns), [1] edge_row6 normalized [0,1], [2] edge_row7 normalized [0,1],
//   [3..19] ch192-207 (squashed perception CfC state, see finding below).
//
// # P1.8 — critical finding: ch192-207 aliased into the NEXT being (fixed)
//
// `output` has EXACTLY `WAVE_U32S=48` words (192 bytes) per being — the same
// `SENSORY_WAVE_SIZE=192` that `brain.wgsl` reads (`const WAVE: u32 = 192u;`, with no
// slack). Writing to "ch192-207" via `write_channel` (which does `ch/4` without a bounds
// check) landed on index `being*48 + 48..51` — the START of the NEXT being's block,
// corrupting (via atomicOr) its channels 0-15 (FAR/NEAR pool, actually read
// by L1). Since `brain.wgsl` never reads an index ≥ 192 (`WAVE=192u`, confirmed by
// grep), the values never even influenced the network — the observable effect was
// ONLY the corruption of the neighboring being. Fix: the 16 values now go to `extra_out`
// (each being's own space, no collision) instead of `output`. The true recurrent
// state (used by the NEXT call of this kernel) remains correct and
// untouched in `cfc_state` (binding 3) — this only affects the extra EXPOSURE channel
// that was never read by anyone in the training pipeline (gpu_evolve).
//
// # Dispatch
//
// `@workgroup_size(1)`, `dispatch_workgroups(n_beings, 1, 1)` — 1 thread per being,
// mirroring `perception_temporal.wgsl` (a pattern proven safe against the ACO/RADV
// compiler SIGSEGV that loops over storage buffers with workgroup_size>1 used to trigger).

struct Params {
    n_beings: u32,
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
}

@group(0) @binding(0) var<uniform>             params:      Params;
@group(0) @binding(1) var<storage, read>       retina:      array<u32>;
@group(0) @binding(2) var<storage, read>       cfc_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> cfc_state:   array<f32>;
@group(0) @binding(4) var<storage, read_write> output:      array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> extra_out:   array<f32>;

const RETINA_ROWS:  u32 = 14u;
const RETINA_COLS:  u32 = 126u;
const RETINA_TOTAL: u32 = 1764u;
const WAVE_U32S:    u32 = 48u;
// P1.8: `extra_out` stride per being — 3 historical values (min_wall_ahead,
// edge_row6, edge_row7) + 16 of the CfC state (ch192-207) that previously aliased in
// `output`. See the finding in the file header (binding 5).
const EXTRA_STRIDE: u32 = 19u;

const POOL_BINS:    u32 = 16u;   // PERCEPTION_BINS
const CFC_HIDDEN:   u32 = 16u;   // CFC_HIDDEN (math/cfc_perception.rs)
const CFC_INPUT:    u32 = 48u;   // CFC_INPUT  (3 zones × 16 bins)

// Offsets inside `cfc_weights` — mirrors `math::cfc_perception::flat_weights()`.
const OFF_W_XI: u32 = 0u;                                   // 16×48 = 768
const OFF_W_HI: u32 = OFF_W_XI + CFC_HIDDEN * CFC_INPUT;     // 768
const OFF_W_F:  u32 = OFF_W_HI + CFC_HIDDEN * CFC_HIDDEN;    // 1024
const OFF_B_I:  u32 = OFF_W_F  + CFC_HIDDEN * CFC_HIDDEN;    // 1280
const OFF_B_F:  u32 = OFF_B_I  + CFC_HIDDEN;                 // 1296

const VOXEL_WALL:    u32 = 2u;
// Central columns (middle third of the 126 pixels) — mirrors context.rs:823-824.
const WALL_COL_LO:   u32 = RETINA_COLS / 3u;       // 42
const WALL_COL_HI:   u32 = 2u * RETINA_COLS / 3u;  // 84
// "No wall ahead" sentinel — only needs to be >= WALL_NEAR_M (3.0 m, in
// statics.rs) for the CPU consumer (`if min_wall_ahead < WALL_NEAR_M`) to
// behave like the `f32::MAX` that Rust used. It does not need to be the exact MAX.
const NO_WALL_SENTINEL: f32 = 1.0e9;

// ─── legacy_encoding — LITERAL COPY of perception.wgsl:251 ───────────────────────
fn legacy_encoding(d: f32, mat: u32) -> i32 {
    let depth_fade = clamp(d / 1200.0, 0.0, 1.0);
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
    } else if mat == 3u { // Hole
        v = 90.0 - (clamp(d / 50.0, 0.0, 1.0) * 30.0);
    } else { // Being (4), `--retina-beings`: flat -29, the one value no other material
             // reaches at any distance. Was falling through to the Hole branch here while
             // `gpu::legacy_encoding` and `fly_map.wgsl` already had the branch — exactly the
             // drift the header of this copy warns about (14/09/2026).
        v = -29.0;
    }
    return i32(clamp(v, -128.0, 127.0));
}

// ─── write_channel — LITERAL COPY of perception.wgsl:208 ─────────────────────────
// P1.8: no call sites in this file since the fix for "ch192-207 aliases into the next
// being" (finding in the header) — `output`/binding 4 stays declared only because the
// bind group layout is shared with `perception.rs`. Kept (not deleted) in case a
// future consumer genuinely needs to write to `output` from here.
fn write_channel(being: u32, ch: u32, val: i32) {
    let clamped  = clamp(val, -128, 127);
    let byte_val = u32(clamped & 0xFF);
    let base     = being * WAVE_U32S + ch / 4u;
    atomicOr(&output[base], byte_val << ((ch % 4u) * 8u));
}

fn read_retina_pixel(being: u32, row: u32, col: u32) -> u32 {
    return retina[being * RETINA_TOTAL + row * RETINA_COLS + col];
}

fn pixel_dist_m(packed: u32) -> f32 { return f32(packed >> 8u) / 1000.0; }
fn pixel_mat(packed: u32)   -> u32 { return packed & 0xFFu; }

// sigmoid — same operation order as `iara_cfc::sigmoid`: `1.0/(1.0+(-x).exp())`.
fn sigmoid_(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wgid: vec3<u32>) {
    let being = wgid.x;
    if being >= params.n_beings { return; }

    // ── 1) ch192-207 uses the OLD STATE (h from before this tick) ─────────────────
    // Preserves the 1-tick lag that already existed (see the file doc). Reads BEFORE
    // touching this tick's retina.
    let sb = being * CFC_HIDDEN;
    var h: array<f32, 16>;
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        h[i] = cfc_state[sb + i];
    }
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        // x/(1+|x|) approximation — LITERAL COPY of perception_temporal.wgsl, block
        // "CfC de percepção" (name still in Portuguese there). Do NOT swap for // check-language: allow
        // native tanh(): it would change the channel value.
        let hi = h[i];
        let abs_h = abs(hi);
        let tanh_h = select(hi / (1.0 + abs_h), sign(hi) * 0.999, abs_h > 10.0);
        let val = i32(clamp(tanh_h * 127.0 * 0.3, -127.0, 127.0));
        // P1.8: this was `write_channel(being, 192u + i, val)` — it aliased into the
        // next being (see finding in the header). Goes to `extra_out`, this being's
        // own space, slots [3..19).
        extra_out[being * EXTRA_STRIDE + 3u + i] = f32(val);
    }

    // ── 2) Sweeps this tick's physical retina: col-pool, min_wall_ahead, edge_row6/7 ──
    var pooled: array<f32, 224>; // 14 rows × 16 bins, row-major (row*16+bin)
    var min_wall_ahead: f32 = NO_WALL_SENTINEL;
    var edge_sum6: i32 = 0;
    var edge_sum7: i32 = 0;
    var edge_prev6: i32 = 0;
    var edge_prev7: i32 = 0;

    for (var row = 0u; row < RETINA_ROWS; row++) {
        let track_edge = row == 6u || row == 7u;
        for (var bin = 0u; bin < POOL_BINS; bin++) {
            let cols_per_bin: u32 = RETINA_COLS / POOL_BINS; // 7
            let lo = bin * cols_per_bin;
            let hi_col: u32 = select(lo + cols_per_bin, RETINA_COLS, bin == POOL_BINS - 1u);
            var mx: i32 = -128;
            for (var col = lo; col < hi_col; col++) {
                let packed = read_retina_pixel(being, row, col);
                let mat    = pixel_mat(packed);
                let dist_m = pixel_dist_m(packed);
                let v = legacy_encoding(dist_m, mat);
                if v > mx { mx = v; }

                if mat == VOXEL_WALL && col >= WALL_COL_LO && col < WALL_COL_HI {
                    min_wall_ahead = min(min_wall_ahead, dist_m);
                }

                // Edge density: sum of |difference| between CONSECUTIVE columns
                // (0..124 vs 1..125) — mirrors context.rs:872-880. The bins cover
                // 0..126 in increasing order without overlap, so iterating
                // `col` inside increasing `bin` reconstructs the full 0..125
                // sequence; `edge_prev` carries the previous pixel's value across
                // the boundary between bins.
                if track_edge {
                    if row == 6u {
                        if col > 0u { edge_sum6 += abs(v - edge_prev6); }
                        edge_prev6 = v;
                    } else {
                        if col > 0u { edge_sum7 += abs(v - edge_prev7); }
                        edge_prev7 = v;
                    }
                }
            }
            pooled[row * POOL_BINS + bin] = f32(mx + 128);
        }
    }

    extra_out[being * EXTRA_STRIDE + 0u] = min_wall_ahead;
    extra_out[being * EXTRA_STRIDE + 1u] = clamp(f32(edge_sum6) / 6000.0, 0.0, 1.0);
    extra_out[being * EXTRA_STRIDE + 2u] = clamp(f32(edge_sum7) / 6000.0, 0.0, 1.0);

    // ── 3) Zone-pool: far=[0,5) mean, mid=[5,10) mean, near=[10,14) max ────────────
    // Mirrors `retina_zones()` + `ZoneConfig::pool` — order far,mid,near in x[48].
    var x: array<f32, 48>;

    // far: rows [0,5), MEAN
    for (var c = 0u; c < POOL_BINS; c++) {
        var acc: f32 = 0.0;
        for (var r = 0u; r < 5u; r++) {
            acc += pooled[r * POOL_BINS + c] / 127.5 - 1.0;
        }
        x[c] = acc * (1.0 / 5.0);
    }
    // mid: rows [5,10), MEAN
    for (var c = 0u; c < POOL_BINS; c++) {
        var acc: f32 = 0.0;
        for (var r = 5u; r < 10u; r++) {
            acc += pooled[r * POOL_BINS + c] / 127.5 - 1.0;
        }
        x[POOL_BINS + c] = acc * (1.0 / 5.0);
    }
    // near: rows [10,14), MAX
    for (var c = 0u; c < POOL_BINS; c++) {
        var mxv: f32 = -1.0e30;
        for (var r = 10u; r < 14u; r++) {
            let v = pooled[r * POOL_BINS + c] / 127.5 - 1.0;
            if v > mxv { mxv = v; }
        }
        x[2u * POOL_BINS + c] = mxv;
    }

    // ── 4) CfC step — mirrors `iara_cfc::CfcCell::step` (see math/cfc_perception.rs) ──
    // matvec with a sequential accumulator, SAME order as `iara_cfc::matvec`.
    var pre_xi: array<f32, 16>;
    for (var r = 0u; r < CFC_HIDDEN; r++) {
        var acc: f32 = 0.0;
        for (var c = 0u; c < CFC_INPUT; c++) {
            acc += cfc_weights[OFF_W_XI + r * CFC_INPUT + c] * x[c];
        }
        pre_xi[r] = acc;
    }
    var pre_hi: array<f32, 16>;
    for (var r = 0u; r < CFC_HIDDEN; r++) {
        var acc: f32 = 0.0;
        for (var c = 0u; c < CFC_HIDDEN; c++) {
            acc += cfc_weights[OFF_W_HI + r * CFC_HIDDEN + c] * h[c];
        }
        pre_hi[r] = acc;
    }
    var pre_f: array<f32, 16>;
    for (var r = 0u; r < CFC_HIDDEN; r++) {
        var acc: f32 = 0.0;
        for (var c = 0u; c < CFC_HIDDEN; c++) {
            acc += cfc_weights[OFF_W_F + r * CFC_HIDDEN + c] * h[c];
        }
        pre_f[r] = acc;
    }

    var h_new: array<f32, 16>;
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        let b_i = cfc_weights[OFF_B_I + i];
        let b_f = cfc_weights[OFF_B_F + i];
        let i_gate = tanh(pre_xi[i] + pre_hi[i] + b_i);
        let f_gate = sigmoid_(pre_f[i] + b_f);
        let decay  = exp(-f_gate);
        h_new[i] = h[i] * decay + i_gate * (1.0 - decay);
    }

    // ── 5) Persists h_new for the NEXT tick ───────────────────────────────────
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        cfc_state[sb + i] = h_new[i];
    }
}
