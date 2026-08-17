// ── EQUIPOTENT CORTICAL SHEET — KERNEL C: MOTOR PROJECTION ───────────────────
//
// Projects the sheet's activations to [throttle, steer, brake].
//
// PRINCIPLE: which units project to the motor is NOT chosen by hand.
// We use the buf_mw weights (motor weights), which start at randomly
// distributed zeros and are learned/evolved. The units with the largest
// |accumulated weight| automatically dominate the motor output.
//
// buf_mw: f32[SHEET_N × 3] per being. Layout: ser*SHEET_N*3 + unit*3 + motor_idx.
// motor_idx: 0=throttle, 1=steer, 2=brake.
//
// Output: buf_motor[ser*3 + idx] = tanh(Σ_i mw[i,idx] * act[i])

const SHEET_N:   u32 = 1024u;
const N_MOTOR:   u32 = 3u;  // throttle=0, steer=1, brake=2

struct Params {
    n_beings: u32,
    tick:     u32,
    _pad0:    u32,
    _pad1:    u32,
}

@group(0) @binding(0) var<uniform>             params:    Params;
// Sheet activations: f32 per unit per being. Layout: ser*SHEET_N + unit.
@group(0) @binding(1) var<storage, read>       buf_act:   array<f32>;
// Motor projection weights: f32. Layout: ser*SHEET_N*3 + unit*3 + motor_idx.
@group(0) @binding(2) var<storage, read>       buf_mw:    array<f32>;
// Motor output: f32[3] per being. Layout: ser*3 + motor_idx.
@group(0) @binding(3) var<storage, read_write> buf_motor: array<f32>;

// LDS for the parallel reduction of the 3 motors
var<workgroup> wg_motor: array<f32, 3>;

// ── Kernel C: 1 workgroup per being, 64 threads ───────────────────────────────
// Each thread accumulates SHEET_N/64 units for the 3 motors.

@compute @workgroup_size(64, 1, 1)
fn motor(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being: u32 = wgid.x;
    let tid:   u32 = lid.x;

    if being >= params.n_beings { return; }

    // Initializes accumulators in LDS (tid 0)
    if tid == 0u {
        wg_motor[0] = 0.0;
        wg_motor[1] = 0.0;
        wg_motor[2] = 0.0;
    }
    workgroupBarrier();

    // Local accumulation per thread (SHEET_N/64 = 16 units)
    var acc_t: f32 = 0.0;  // throttle
    var acc_s: f32 = 0.0;  // steer
    var acc_b: f32 = 0.0;  // brake

    let act_base: u32 = being * SHEET_N;
    let mw_base:  u32 = being * SHEET_N * N_MOTOR;

    for (var i: u32 = tid * 16u; i < (tid + 1u) * 16u; i = i + 1u) {
        if i >= SHEET_N { break; }
        let a: f32 = buf_act[act_base + i];
        let row: u32 = mw_base + i * N_MOTOR;
        acc_t = acc_t + buf_mw[row + 0u] * a;
        acc_s = acc_s + buf_mw[row + 1u] * a;
        acc_b = acc_b + buf_mw[row + 2u] * a;
    }

    // Reduction via thread 0 (simple enough — 64 accumulations)
    // On RDNA2 / WGSL, without native atomicAdd<f32>, we use tid 0 as the master.
    if tid == 0u {
        var sum_t: f32 = 0.0;
        var sum_s: f32 = 0.0;
        var sum_b: f32 = 0.0;
        for (var j: u32 = 0u; j < SHEET_N; j = j + 1u) {
            let a: f32  = buf_act[act_base + j];
            let row: u32 = mw_base + j * N_MOTOR;
            sum_t = sum_t + buf_mw[row + 0u] * a;
            sum_s = sum_s + buf_mw[row + 1u] * a;
            sum_b = sum_b + buf_mw[row + 2u] * a;
        }
        // tanh to saturate the output in (-1, 1)
        let motor_base: u32 = being * N_MOTOR;
        buf_motor[motor_base + 0u] = tanh(sum_t);
        buf_motor[motor_base + 1u] = tanh(sum_s);
        buf_motor[motor_base + 2u] = tanh(sum_b);
    }
}
