// ── SLOW BRAIN ON THE GPU ─────────────────────────────────────────────────────
//
// Exact port of SlowBrain (engine.rs) + CfCCell (cfc.rs) to WGSL.
//
// # Pipeline per being (1 workgroup of 64 threads)
//
//   input[48] i8 → zero padding → cfc_input[64]
//   → CfC(64, resident state) → cfc_out[64] i8 = tanh(state)*127
//   → f32: cfc_f32[i] = cfc_out[i] / 127.0
//   → Dense1(64→16) with f32 weights → relu → d1[16]
//   → Dense2(16→3) with f32 weights → tanh → output[3]
//
//   Runs 1 tick out of every 8 (tick_interval = 8). On skipped ticks, output PERSISTS.
//
// # Buffer layout (SoA)
//
//   buf_weights:  dense1.weights(1024) | dense1.bias(16) | dense2.weights(48) | dense2.bias(3)
//                 = 1091 f32 per being. Order matches SlowBrain::all_weights().
//
//   buf_state:    per being: [cfc_state[64], cfc_fatigue[64], counter_u32_bitcast_f32,
//                           tick_counter_u32_bitcast_f32, output[3]]
//                 = 64 + 64 + 1 + 1 + 3 = 133 f32 per being
//
//   buf_input:    [48 i32 per being]  (sign-extended from i8 by the host)
//
//   buf_output:   [3 f32 per being]   (throttle, steer, brake) — result per tick
//
// # SMALL kernel (RADV/ACO)
//
// Two kernels chained by the host in the same CommandEncoder:
//   Kernel A (entry point: run_cfc) — CfC stage: input→state→cfc_out
//   Kernel B (entry point: run_dense) — Dense1+Dense2→output stage
//
// Splitting into two kernels was the choice that keeps each shader < 100 body
// lines, within the limit RADV/ACO compiled without crashing in this project.
//
// # CfC — identical to cfc.rs
//
//   elapsed_dt = body_dt * tick_interval
//   decay = exp(-0.5 * elapsed_dt)
//   gain_dt = 2.0 * (1.0 - decay)
//   x_raw = input_i32 / 127.0
//   threshold = FATIGUE_BETA * fatigue[i]
//   x = sign(x_raw) * max(0, |x_raw| - threshold)    ← soft shrinkage
//   s = s * decay + x * gain_dt
//   if h % 7 == 0: s += noise  (Knuth hash, ~14% of the channels)
//   s = clamp(s, -2.0, 2.0)
//   fatigue[i] += FATIGUE_ALPHA * (|s| - fatigue[i]) * dt
//   cfc_out[i] = i32(clamp(tanh(s) * 127, -127, 127))
//
// FATIGUE_ALPHA = 1.0, FATIGUE_BETA = 0.4 (same values as cfc.rs)

const CFC_SIZE:      u32 = 64u;
const INPUT_SIZE:    u32 = 48u;   // real input; 48..63 are zeros
const D1_IN:         u32 = 64u;   // Dense1 input = CFC_SIZE
const D1_OUT:        u32 = 16u;   // Dense1 output
const D2_IN:         u32 = 16u;   // Dense2 input = D1_OUT
const D2_OUT:        u32 = 3u;    // Dense2 output (throttle, steer, brake)

// buf_state slots per being (offsets in f32):
//   [0..64)   cfc_state
//   [64..128) cfc_fatigue
//   [128]     counter (u32 bitcast)
//   [129]     tick_counter (u32 bitcast)
//   [130..133) output[3]  (persists between ticks)
const STATE_PER_SER:    u32 = 133u;
const OFF_CFC_STATE:    u32 = 0u;
const OFF_CFC_FATIGUE:  u32 = 64u;
const OFF_COUNTER:      u32 = 128u;
const OFF_TICK_COUNTER: u32 = 129u;
const OFF_OUTPUT:       u32 = 130u;

// Weights per being in buf_weights:
//   [0..1024)    dense1.weights  (64×16, row-major: weights[j*64+i])
//   [1024..1040) dense1.bias     (16)
//   [1040..1088) dense2.weights  (16×3, row-major: weights[j*16+i])
//   [1088..1091) dense2.bias     (3)
const WEIGHTS_PER_SER:  u32 = 1091u;
const OFF_D1W:          u32 = 0u;
const OFF_D1B:          u32 = 1024u;   // 64*16 = 1024
const OFF_D2W:          u32 = 1040u;   // 1024 + 16
const OFF_D2B:          u32 = 1088u;   // 1040 + 48

// CfC parameters (same as cfc.rs and brain.wgsl)
const CFC_DECAY_RATE:  f32 = 0.5;
const CFC_DC_GAIN:     f32 = 2.0;
const CFC_CLAMP:       f32 = 2.0;
const CFC_DENOISE_MAG: f32 = 0.10;
const FATIGUE_ALPHA:   f32 = 1.0;
const FATIGUE_BETA:    f32 = 0.4;

// tick_interval: runs 1× every 8 ticks (engine.rs: tick_interval = 8)
const TICK_INTERVAL:   u32 = 8u;

// ── UNIFORM ───────────────────────────────────────────────────────────────────

struct Params {
    n_beings: u32,
    dt:       f32,
    _pad0:    u32,
    _pad1:    u32,
}

// ── BINDINGS (same for both kernels) ──────────────────────────────────────────

@group(0) @binding(0) var<uniform>             params:      Params;

// input: 48 i32 per being (sign-extended from i8 by the host)
// Layout: being * INPUT_SIZE + channel
@group(0) @binding(1) var<storage, read>       buf_input:   array<i32>;

// Dense1 + Dense2 weights (immutable during a tick; uploaded per generation/spawn)
// Layout: being * WEIGHTS_PER_SER + local_offset
@group(0) @binding(2) var<storage, read>       buf_weights: array<f32>;

// resident state (cfc_state, cfc_fatigue, counter, tick_counter, output)
// Layout: being * STATE_PER_SER + local_offset
@group(0) @binding(3) var<storage, read_write> buf_state:   array<f32>;

// final output [3 f32 per being] for fast host reads
// Written by kernel B (run_dense) only on the ticks that run.
// On skipped ticks the host reads buf_state[OFF_OUTPUT..] directly,
// but keeping this separate buffer allows copy_buffer_to_buffer without arithmetic.
// Layout: being * 3 + {0,1,2}
@group(0) @binding(4) var<storage, read_write> buf_output:  array<f32>;

// ── WORKGROUP MEMORY (shared among the 64 threads of the same being) ──────────

// CfC output: tanh(state)*127 as i32, available to Dense1.
// Produced by kernel A, consumed by kernel B (via the state buffer, not LDS,
// because the two kernels are separate dispatches — LDS does not persist between them).
// Within each kernel, we use workgroup vars for intra-kernel communication.

// Kernel A: w_cfc_out holds the CfC result so the host can read it via
// buf_state if needed, but here it serves kernel B.
var<workgroup> w_cfc_out: array<i32, 64>;

// Kernel B: w_d1_raw holds Dense1's linear output (before the relu).
var<workgroup> w_d1_raw: array<f32, 64>;  // 64 slots, only D1_OUT=16 used

// ── KERNEL A: CfC ─────────────────────────────────────────────────────────────
//
// 64 threads, one per CfC channel.
// Reads buf_input[0..48] (zeros at 48..63), updates state and fatigue,
// writes cfc_out into buf_state[OFF_CFC_STATE..] as f32/127 (to be read
// by kernel B as f32 instead of redoing the division).
//
// IMPORTANT: also updates tick_counter. If the tick is skipped, it writes
// the incremented counter and returns WITHOUT touching the output.
// Kernel B checks the tick_counter to decide whether to run.

@compute @workgroup_size(64)
fn run_cfc(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being = wgid.x;
    let tid   = lid.x;   // [0, 63]

    if being >= params.n_beings { return; }

    let state_base:   u32 = being * STATE_PER_SER;
    let input_base:   u32 = being * INPUT_SIZE;

    // Reads and increments tick_counter (only tid==0 writes, all read the current value).
    let tc_old: u32 = bitcast<u32>(buf_state[state_base + OFF_TICK_COUNTER]);
    let tc_new: u32 = tc_old + 1u;
    if tid == 0u {
        buf_state[state_base + OFF_TICK_COUNTER] = bitcast<f32>(tc_new);
    }

    // If this tick is not a cortex tick, there is nothing to do.
    // The previous output persists in buf_state[OFF_OUTPUT] and buf_output.
    if (tc_new % TICK_INTERVAL) != 0u { return; }

    // ── Updates the denoising counter (independent of tick_counter) ───────────
    let cnt_old: u32 = bitcast<u32>(buf_state[state_base + OFF_COUNTER]);
    let cnt_new: u32 = cnt_old + 1u;
    if tid == 0u {
        buf_state[state_base + OFF_COUNTER] = bitcast<f32>(cnt_new);
    }

    // ── Reads channel tid's state ─────────────────────────────────────────────
    let i: u32 = tid;
    var s:   f32 = buf_state[state_base + OFF_CFC_STATE   + i];
    var fat: f32 = buf_state[state_base + OFF_CFC_FATIGUE + i];

    // This organ runs one update for every eight body ticks. Using one body `dt`
    // here stretched every time constant by 8x and made GPU memory disagree with
    // `SlowBrain::tick`. The current input is a zero-order-held sample covering the
    // elapsed interval.
    let dt: f32     = params.dt * f32(TICK_INTERVAL);
    let decay: f32  = exp(-CFC_DECAY_RATE * dt);
    let gain_dt:f32 = CFC_DC_GAIN * (1.0 - decay);

    // Channel i input: channels 0..47 come from buf_input; channels 48..63 are zeros.
    var x_raw: f32;
    if i < INPUT_SIZE {
        x_raw = f32(buf_input[input_base + i]) / 127.0;
    } else {
        x_raw = 0.0;
    }

    // Soft shrinkage — mirrors cfc.rs exactly
    let threshold: f32 = FATIGUE_BETA * fat;
    let x: f32 = sign(x_raw) * max(0.0, abs(x_raw) - threshold);

    // CfC equation
    s = s * decay + x * gain_dt;

    // Deterministic noise via Knuth hash (~14% of the channels)
    let h: u32 = cnt_new * 2654435761u + i;
    if (h % 7u) == 0u {
        let noise: f32 = (f32(h >> 8u) / 16777216.0 - 0.5) * 2.0 * CFC_DENOISE_MAG;
        s = s + noise;
    }

    s = clamp(s, -CFC_CLAMP, CFC_CLAMP);

    // Updates fatigue
    fat = fat + FATIGUE_ALPHA * (abs(s) - fat) * dt;

    // Saves the updated state
    buf_state[state_base + OFF_CFC_STATE   + i] = s;
    buf_state[state_base + OFF_CFC_FATIGUE + i] = fat;

    // Saving cfc_out as normalized f32 [-1,1] in temporary state
    // (using OFF_OUTPUT+3..67 would be wrong — we keep it in w_cfc_out via workgroup)
    // Since kernel B is a separate dispatch, we communicate via buf_state.
    // The OFF_CFC_STATE slots are reserved for state, but kernel B needs cfc_f32.
    // Solution: kernel B redoes tanh(s)/1 — that is, reads s from buf_state and applies tanh.
    // This avoids an extra buffer and keeps the code small.
    // (cfc_out = tanh(s), which kernel B will recompute by reading s from buf_state)
}

// ── KERNEL B: Dense1 + Dense2 ─────────────────────────────────────────────────
//
// 64 threads. Only D1_OUT=16 threads active for Dense1; D2_OUT=3 for Dense2.
// Threads 16..63 stay idle — ok for RDNA2 (2 waves of 32, one partial).
//
// Flow:
//   1. Checks tick_counter: if not a cortex tick, copies buf_state→buf_output and returns.
//   2. Recomputes cfc_f32[i] = tanh(buf_state[CFC_STATE+i]) for i in [0,64).
//   3. Dense1: 16 threads compute w_d1_raw[j] = Σ weights[j*64+i] * cfc_f32[i] + bias[j]
//   4. Inline ReLU: d1[j] = max(0, w_d1_raw[j])
//   5. Dense2: 3 threads compute output[k] = tanh(Σ weights[k*16+j] * d1[j] + bias[k])
//   6. Saves output into buf_state[OFF_OUTPUT..] and buf_output[being*3..].

var<workgroup> w_cfc_f32: array<f32, 64>;
var<workgroup> w_d1_relu: array<f32, 16>;

@compute @workgroup_size(64)
fn run_dense(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being = wgid.x;
    let tid   = lid.x;

    if being >= params.n_beings { return; }

    let state_base:   u32 = being * STATE_PER_SER;
    let weight_base:  u32 = being * WEIGHTS_PER_SER;
    let out_base:     u32 = being * D2_OUT;

    // Checks whether this is a cortex tick (tick_counter was already incremented by kernel A)
    let tc: u32 = bitcast<u32>(buf_state[state_base + OFF_TICK_COUNTER]);
    if (tc % TICK_INTERVAL) != 0u {
        // Skipped tick: copies the persisted output from the state into buf_output.
        // Only 3 threads needed.
        if tid < D2_OUT {
            buf_output[out_base + tid] = buf_state[state_base + OFF_OUTPUT + tid];
        }
        return;
    }

    // ── Stage 1: recomputes cfc_f32 (tanh of the state updated by kernel A) ───
    if tid < CFC_SIZE {
        let s: f32 = buf_state[state_base + OFF_CFC_STATE + tid];
        // The CPU does a round-trip through i8 between the CfC and Dense1:
        //   output[i] = (tanh(s) * 127).clamp(-127,127) as i8   // `as i8` TRUNCATES
        //   cfc_f32[i] = output[i] as f32 / 127
        // This quantizes in steps of 1/127. Using `tanh(s)` directly looks harmless
        // (error ≤ 0.004 per channel) but there are 64 channels summed with weights ~0.35 — measured:
        // mean error 2.9e-3 and worst 1.2e-2 at the output, a hundred times the tolerance.
        // The quantization is part of the equation, not a storage detail.
        let q: f32 = clamp(tanh(s) * 127.0, -127.0, 127.0);
        w_cfc_f32[tid] = trunc(q) / 127.0;
    }

    workgroupBarrier();

    // ── Stage 2: Dense1 (64→16) ───────────────────────────────────────────────
    // Thread tid ∈ [0,15] computes Dense1's output neuron j.
    if tid < D1_OUT {
        let j: u32 = tid;
        let w_base: u32 = weight_base + OFF_D1W + j * D1_IN;
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < D1_IN; i = i + 1u) {
            // anti-FMA: separate product before summing (consistency with the CPU)
            let prod: f32 = buf_weights[w_base + i] * w_cfc_f32[i];
            acc = acc + prod;
        }
        acc = acc + buf_weights[weight_base + OFF_D1B + j];  // bias
        // ReLU
        w_d1_relu[j] = max(0.0, acc);
    }

    workgroupBarrier();

    // ── Stage 3: Dense2 (16→3) → tanh ────────────────────────────────────────
    // Thread tid ∈ [0,2] computes output k.
    if tid < D2_OUT {
        let k: u32 = tid;
        let w_base: u32 = weight_base + OFF_D2W + k * D2_IN;
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < D2_IN; j = j + 1u) {
            let prod: f32 = buf_weights[w_base + j] * w_d1_relu[j];
            acc = acc + prod;
        }
        acc = acc + buf_weights[weight_base + OFF_D2B + k];  // bias
        let out: f32 = tanh(acc);

        // Saves in two places:
        //   buf_state: persists for future ticks (output hold)
        //   buf_output: read by the host after readback
        buf_state[state_base + OFF_OUTPUT + k] = out;
        buf_output[out_base  + k]              = out;
    }
}
