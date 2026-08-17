// ── EQUIPOTENT CORTICAL SHEET — KERNEL B: HEBBIAN PLASTICITY ─────────────────
//
// Updates the afference weights buf_afw via the Hebb rule with subtractive
// normalization (competition for a finite resource):
//
//   Δw_ik = η · (a_i · a_k − α · w_ik · Σ_j a_j · a_k)
//
// Subtractive normalization makes correlated inputs strengthen the connection
// and decorrelated inputs weaken it — the boundary emerges where the
// cross-correlation drops (doc §4).
//
// Dispatch: (SHEET_N, n_beings, 1) — 1 thread per unit per being.
// workgroup_size(32) → each WG handles 32 units of the same being.

const SHEET_W:  u32 = 32u;
const SHEET_H:  u32 = 32u;
const SHEET_N:  u32 = 1024u;
const N_INPUTS: u32 = 223u;

// Hebb learning rate. Small → gradual change tick by tick.
const ETA:   f32 = 0.005;
// Subtractive normalization coefficient (competition for a finite resource).
const ALPHA: f32 = 0.003;
// Weight bounds
const W_MAX: f32 =  5.0;
const W_MIN: f32 = -1.0;

struct Params {
    n_beings:   u32,
    tick:       u32,
    // act_sum: Σ_j |act_j| precomputed by the CPU and passed here to avoid an
    // intra-kernel reduction that could cause race conditions.
    // The CPU reads buf_act after kernel A and sums the 1024 values (trivial, 1024 f32).
    act_sum:    f32,
    _pad0:      u32,
}

@group(0) @binding(0) var<uniform>             params:   Params;
// Sheet activation (kernel A output). Layout: being*SHEET_N + unit.
@group(0) @binding(1) var<storage, read>       buf_act:  array<f32>;
// Sensory signal. Layout: being*N_INPUTS + channel.
@group(0) @binding(2) var<storage, read>       buf_wave: array<i32>;
// Hebbian afference weights. Layout: being*SHEET_N*N_INPUTS + unit*N_INPUTS + channel.
@group(0) @binding(3) var<storage, read_write> buf_afw:  array<f32>;

// ── Kernel B: 1 thread per unit, dispatch (SHEET_N/32, n_beings, 1) ───────────

@compute @workgroup_size(32, 1, 1)
fn hebb(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let unit:  u32 = gid.x;   // index in the sheet [0, SHEET_N)
    let being: u32 = gid.y;   // being index

    if unit >= SHEET_N || being >= params.n_beings { return; }

    let a_i:       f32 = buf_act[being * SHEET_N + unit]; // activation of unit i
    let wave_base: u32 = being * N_INPUTS;
    let afw_row:   u32 = being * SHEET_N * N_INPUTS + unit * N_INPUTS;
    let act_sum:   f32 = params.act_sum; // Σ_j |act_j| precomputed by the CPU

    // Δw_ik = η · (a_i · a_k − α · w_ik · |a_k| · act_sum)
    for (var k: u32 = 0u; k < N_INPUTS; k = k + 1u) {
        let a_k:    f32 = f32(buf_wave[wave_base + k]) / 127.0;
        let w_old:  f32 = buf_afw[afw_row + k];

        let hebb_term: f32 = a_i * a_k;
        // Subtractive normalization: proportional to the current weight and total activity
        let norm_term: f32 = ALPHA * w_old * act_sum * abs(a_k);
        let delta_w:   f32 = ETA * (hebb_term - norm_term);

        buf_afw[afw_row + k] = clamp(w_old + delta_w, W_MIN, W_MAX);
    }
}
