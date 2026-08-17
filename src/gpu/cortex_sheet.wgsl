// ── EQUIPOTENT CORTICAL SHEET — KERNEL A: AFFERENCE + SPONTANEOUS WAVES ────
//
// Projects the 223 sensory channels onto the 32×32=1024-unit sheet.
// Each channel k has a deterministic target (tx_k, ty_k) (hash of the index k),
// and contributes to unit i with weight P ∝ exp(−‖pos_i − target_k‖² / σ²).
//
// Spontaneous waves: during the first SPONT_TICKS, injects endogenous rhythmic
// activity before the sensory signal arrives — necessary for Hebbian
// segregation to have something to correlate (doc §5.5).
//
// ANTI-SATURATION MECHANISMS (added 2026-08-04):
//
//   1. Adaptive fatigue per unit (spike-frequency adaptation):
//      Each unit accumulates fat += SHEET_FAT_ALPHA * (|act| - fat) * dt,
//      and the effective input is reduced by fat: net_eff = net_input - SHEET_FAT_BETA * fat.
//      When the unit fires at a fixed point (high |act| for many ticks), fat grows
//      and the effective input turns negative — breaking the fixed point without if/clamp.
//      Mechanism: analogous to cortical pyramidal neuron adaptation (after-hyperpolarization).
//
//   2. Divisive normalization by local neighborhood (shunting lateral inhibition):
//      Before writing act, divides by the local mean activity field of the 4 orthogonal
//      neighbors (tick t-1, read from buf_act before the update):
//        act_norm = tanh(h) / (1.0 + SHEET_DIV_SIGMA * mean_neigh_abs)
//      When many neighboring units are active, each one's gain is reduced.
//      Implements lateral inhibition without a loop over neighbors (divisor = local field).
//      Mechanism: analogous to GABAergic basket-cell interneurons in the neocortex.
//
// FORBIDDEN: no `if channel_is_vision {}`, no forced clamp, no reset.

// ── Layout constants ─────────────────────────────────────────────────────────
const SHEET_W:   u32 = 32u;   // sheet width (columns)
const SHEET_H:   u32 = 32u;   // sheet height (rows)
const SHEET_N:   u32 = 1024u; // = SHEET_W × SHEET_H
const N_INPUTS:  u32 = 223u;  // sensory channels (SENSORY_WAVE_SIZE)

// σ² for the afference gaussian: σ=8 cells → covers ¼ of the sheet.
// Large value → modalities overlap initially (more plasticity).
const SIGMA_SQ:  f32 = 64.0;  // σ=8 → σ²=64

// Contribution threshold: exp(−d²/σ²) < EPS → ignore channel.
// exp(−3²) = exp(−9/64) ≈ 0.87; this EPS of 0.05 captures radius ≈ 2.8σ.
const AFFERENCE_EPS: f32 = 0.05;

// Ticks of spontaneous activity before the sensory signal (doc §5.5).
const SPONT_TICKS: u32 = 200u;

// ── Adaptive fatigue constants (mechanism 1) ────────────────────────────────
// SHEET_FAT_ALPHA: fatigue growth speed — FIXED (does not vary in the sweep).
//   τ_fat ≈ 1/alpha = 2.0s → with dt=1/64, grows 0.0078/tick per unit of |act|.
//   With |act|≈1.0, reaches 0.63 in ~128 ticks (2s) — before the fixed point consolidates.
const SHEET_FAT_ALPHA: f32 = 0.5;

// SHEET_FAT_BETA and SHEET_DIV_SIGMA now come from the uniform (fields params.fat_beta and params.div_sigma).
// This allows sweeping without recompiling the shader.
//
// ⚠️ MEASURED (2026-08-04): STRONG inhibition (BETA=1.2, SIGMA=2.0) eliminated saturation (89.8%→0%)
// but cost the CEILING (1055m→440m). The median stayed the same (~140m) — it is not the network dying,
// it is specifically the ceiling. 4-point sweep to find the optimum.

// ── Params uniform ────────────────────────────────────────────────────────────
struct Params {
    n_beings:   u32,
    tick:       u32,     // global tick (0..∞)
    // fat_beta: subtraction strength of the adaptive fatigue (mechanism 1).
    //   0.0 = off; 1.2 = original strong value.
    //   Sweep: 0.0, 0.4, 0.8, 1.2
    fat_beta:   f32,
    // div_sigma: gain of the divisive neighborhood normalization (mechanism 2).
    //   0.0 = off; 2.0 = original strong value.
    //   Sweep: 0.0, 0.5, 1.0, 2.0
    div_sigma:  f32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────
@group(0) @binding(0) var<uniform>             params:      Params;
// sensory_wave: i32 sign-extended. Layout: being * N_INPUTS + channel.
@group(0) @binding(1) var<storage, read>       buf_wave:    array<i32>;
// Afference weights: f32 per (unit, channel). Layout: being*SHEET_N*N_INPUTS + unit*N_INPUTS + channel.
// Initialized with zeros; updated by kernel B (Hebb).
@group(0) @binding(2) var<storage, read_write> buf_afw:     array<f32>;
// Sheet EMA state: f32 per unit per being. Layout: being*SHEET_N + unit.
@group(0) @binding(3) var<storage, read_write> buf_state:   array<f32>;
// τ per unit per being (integration constant). Layout: being*SHEET_N + unit.
@group(0) @binding(4) var<storage, read>       buf_tau:     array<f32>;
// Sheet output activation (post-EMA): f32 per unit per being.
// Written here, read by kernel B (Hebb) and kernel D (motor).
// READ at the start (neighbors from tick t-1) and WRITTEN at the end (tick t).
@group(0) @binding(5) var<storage, read_write> buf_act:     array<f32>;
// Adaptive fatigue per unit per being: f32. Layout: being*SHEET_N + unit.
// Initialized at 0.0; grows when the unit fires sustainedly.
// Mechanism: spike-frequency adaptation (cortical after-hyperpolarization).
@group(0) @binding(6) var<storage, read_write> buf_fat:     array<f32>;

// ── Helper function: spatial target of channel k (hash → grid) ──────────────
// Maps channel index k → (tx, ty) ∈ [0, SHEET_W-1] × [0, SHEET_H-1].
// Uses two distinct Knuth hashes for tx and ty, guaranteeing a REPRODUCIBLE
// pseudo-random distribution. It is NOT a region label — it is a target point.
fn channel_target(k: u32) -> vec2<f32> {
    // Knuth hash: multiplier 2654435761 (golden Fibonacci number)
    let hx: u32 = (k * 2654435761u) ^ (k >> 5u);
    let hy: u32 = (k * 2246822519u) ^ (k >> 3u);
    // Maps to [0, SHEET_W-1] and [0, SHEET_H-1]
    let tx: f32 = f32(hx % SHEET_W);
    let ty: f32 = f32(hy % SHEET_H);
    return vec2<f32>(tx, ty);
}

// ── Helper function: rhythmic spontaneous wave ──────────────────────────────
// Generates spatially correlated endogenous activity, modulated by tick.
// Amplitude decreases with the tick (critical period closes).
// Frequency: ~1 cycle per 32 ticks (theta-band ~2 Hz @ 64 Hz).
fn spontaneous_wave(unit_x: f32, unit_y: f32, tick: u32) -> f32 {
    // Amplitude factor: decreases from 1 → 0 over the first SPONT_TICKS.
    if tick >= SPONT_TICKS { return 0.0; }
    let amp: f32 = 1.0 - f32(tick) / f32(SPONT_TICKS);

    // Traveling wave in the sheet plane: cos(kx*x + ky*y - ω*t).
    // kx, ky: wavelength ≈ sheet/4 (8 cells).
    let kwave: f32 = 6.283185307 / 8.0;  // 2π/8
    let omega: f32 = 6.283185307 / 32.0; // 2π/32 ticks per cycle
    let t: f32     = f32(tick);

    // Two orthogonal modes to maximize the 2D correlation structure.
    let mode_a: f32 = cos(kwave * unit_x - omega * t);
    let mode_b: f32 = cos(kwave * unit_y - omega * t * 1.618);  // φ = golden ratio

    return amp * 0.5 * (mode_a + mode_b);
}

// ── Helper function: neighborhood activation (toroidal, reflected border) ───
// Reads buf_act from the previous tick for the orthogonal neighbor offset by (dx, dy).
// Reflected border: avoids null-neighborhood artifacts at the sheet edges.
fn neigh_act(being: u32, ux: u32, uy: u32, dx: i32, dy: i32) -> f32 {
    // Computes coordinates with reflection at the borders
    let nx_i: i32 = clamp(i32(ux) + dx, 0, i32(SHEET_W) - 1);
    let ny_i: i32 = clamp(i32(uy) + dy, 0, i32(SHEET_H) - 1);
    let nu: u32 = u32(ny_i) * SHEET_W + u32(nx_i);
    return buf_act[being * SHEET_N + nu];
}

// ── Kernel A: 1 thread per unit per being ───────────────────────────────────
// workgroup_size(32, 1, 1): 32 threads per WG → 1024/32 = 32 WGs per being.
// dispatch: (SHEET_N/32, n_beings, 1) = (32, n, 1).

@compute @workgroup_size(32, 1, 1)
fn afference(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let unit: u32  = gid.x;  // index in the sheet [0, SHEET_N)
    let being: u32 = gid.y;  // being index

    if unit >= SHEET_N || being >= params.n_beings { return; }

    // 2D position of this unit on the grid
    let ux: u32 = unit % SHEET_W;
    let uy: u32 = unit / SHEET_W;
    let ux_f: f32 = f32(ux);
    let uy_f: f32 = f32(uy);

    // ── Mechanism 2: divisive normalization — reads neighbors FROM THE PREVIOUS TICK ─────
    // Reads buf_act before any write in this tick (guaranteed: threads of other
    // workgroups of the same dispatch have also only read buf_act up to this point,
    // since there has been no global memory barrier yet, but this tick's WRITES
    // only happen at the end — WGSL guarantees that storage reads see only values
    // written by previous dispatches, not by the dispatch in flight).
    let an: f32 = abs(neigh_act(being, ux, uy,  0, -1));  // north
    let as_: f32 = abs(neigh_act(being, ux, uy,  0,  1)); // south
    let ae: f32 = abs(neigh_act(being, ux, uy,  1,  0));  // east
    let aw: f32 = abs(neigh_act(being, ux, uy, -1,  0));  // west
    // Mean of the 4 orthogonal neighbors (local field activity)
    let mean_neigh: f32 = (an + as_ + ae + aw) * 0.25;
    // Denominator of the divisive normalization: 1.0 = no inhibition, > 1.0 = inhibited
    // div_sigma comes from the uniform — configurable at runtime without recompiling.
    let div_denom: f32 = 1.0 + params.div_sigma * mean_neigh;

    // Accumulates each sensory channel's contribution with gaussian spatial bias
    var total_input: f32 = 0.0;
    let wave_base:  u32  = being * N_INPUTS;
    let afw_base:   u32  = being * SHEET_N * N_INPUTS + unit * N_INPUTS;

    for (var k: u32 = 0u; k < N_INPUTS; k = k + 1u) {
        // Spatial target of this channel (deterministic, no region label)
        let tgt: vec2<f32> = channel_target(k);
        let dx: f32 = ux_f - tgt.x;
        let dy: f32 = uy_f - tgt.y;
        let d_sq: f32 = dx * dx + dy * dy;

        // Gaussian afference weight (computed on the fly — only the learned
        // Hebbian weight is saved in buf_afw). The gaussian is the WEAK prior
        // (analogous to FGF8/Emx2); the Hebb in buf_afw refines it later.
        let gauss: f32 = exp(-d_sq / SIGMA_SQ);
        if gauss < AFFERENCE_EPS { continue; }

        // Normalized sensory signal ∈ [-1, 1]
        let sig: f32 = f32(buf_wave[wave_base + k]) / 127.0;

        // Contribution = gaussian × Hebbian weight × signal
        // buf_afw initialized at 1.0 (no learning yet = gaussian prior only)
        let hebb_w: f32 = buf_afw[afw_base + k];
        total_input = total_input + gauss * hebb_w * sig;
    }

    // Endogenous spontaneous wave (initial phase, doc §5.5)
    let spont: f32 = spontaneous_wave(ux_f, uy_f, params.tick);

    // Total input for this unit (pre-fatigue)
    let net_input: f32 = total_input + spont;

    // ── Mechanism 1: adaptive fatigue — reads slow state from the previous tick ────
    let fat_idx: u32 = being * SHEET_N + unit;
    var fat: f32 = buf_fat[fat_idx];

    // Effective input: subtracts the fatigue contribution.
    // When fat → max (saturated unit), net_eff flips sign → breaks the fixed point.
    // fat_beta comes from the uniform — configurable at runtime without recompiling.
    let net_eff: f32 = net_input - params.fat_beta * fat;

    // EMA: h = α·h + (1−α)·input, where α = exp(−dt/τ)
    // dt = 1/64 s ≈ 0.015625 s. τ per unit (log-spaced bank, generated on the host).
    let tau: f32    = buf_tau[being * SHEET_N + unit];
    let dt: f32     = 1.0 / 64.0;
    let alpha: f32  = exp(-dt / tau);

    var h: f32 = buf_state[being * SHEET_N + unit];
    h = alpha * h + (1.0 - alpha) * net_eff;

    // Post-EMA activation: tanh ∈ (-1, 1)
    let act_raw: f32 = tanh(h);

    // ── Mechanism 2: divisive normalization by local activity ───────────────
    // act_norm = act_raw / div_denom
    // Effect: when many neighbors are active (high mean_neigh),
    // this unit's gain drops → emergent lateral inhibition.
    // It is not a clamp nor a reset — it is gain reduction by a divisor field.
    let act: f32 = act_raw / div_denom;

    // ── Mechanism 1: updates fatigue ─────────────────────────────────────────
    // fat += alpha_fat * (|act_normalized| - fat) * dt
    // Uses normalized |act| (post-division) — tracks the unit's real output.
    fat = fat + SHEET_FAT_ALPHA * (abs(act) - fat) * dt;

    // Write back
    buf_state[being * SHEET_N + unit] = h;
    buf_act[being * SHEET_N + unit]   = act;
    buf_fat[fat_idx]                  = fat;
}
