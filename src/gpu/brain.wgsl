// ── BRAIN ON THE GPU ──────────────────────────────────────────────────────────
//
// # Role in the system
//
// This kernel replaces the NEURAL part of `SovereignEngine::execute_cycle` for every
// being in a single dispatch. The CPU still owns the evolution pipeline, the
// fatigue/metabolism timers, the Trinity (Gate + Reflex + Sentinel) and everything with
// heavy branching over non-tensor state.
//
// What the kernel produces: `final_intent[0..96]` f32 per being, equivalent to the field
// `SovereignEngine::final_intent` BEFORE the Trinity and `axiom_policy`.
//
// # Pipeline per being (one workgroup of 128 threads)
//
//   sensory_wave[WAVE=192] ──► per-channel KAN (16-knot spline)
//                       ──► CfC pre-processor (channels 0..96, resident state)
//                       ──► BitLinear L1  (192→96, f32 weights, i8 input)
//                       ──► tanh × 127 → i8  (quantise for L2)
//                       ──► FloatLinear L2 (96→48, f32 weights)
//                       ──► [optional, params.norm_l2_by_std] / std dev of the 48
//                       ──► final_intent[0..48]  = tanh(l2_raw × params.out_gain)
//                       + l1_raw[48..96] → tanh(× params.out_gain) → final_intent[48..96]
//                       (cortex_l3 is NOT used on the CPU — omitted here)
//
//   `params.out_gain` (default 2.0, identical to the historical hardcode) and
//   `params.norm_l2_by_std` (default 1/ON since P1.8 — see docs/PLANO_EVOLUCAO.md
//   section "P1.8", finding "L2 has to work") are the two parameters of the P1.5 A/B
//   against L2 saturation (48/48 neurons active in 97.4% of the ticks, entropy
//   0.301 nats). See `BrainParams` on the host (`brain.rs`) and STAGE 2 below.
//
//   # P1.8 (revised 2026-08-14) — what each slice of final_intent feeds
//
//   TRAINING (`gpu_evolve`): only [0..3) — throttle/steer/brake through `kan_eval`
//   (context.rs:954-957). The gearbox is automatic by RPM; [3] and [4] are NEVER read in
//   that pipeline (an earlier version of this comment said "0-4 through shift" — that
//   only holds in Bevy). BEVY ENGINE: [0..3) are actuators (motor.rs:2026-2097), [3]/[4]
//   the gearbox (motor.rs:2451-2452), and [0..64) enter myelination/coherence — BUT
//   `myelinate()` has been a deliberate no-op since the F32 migration (layers.rs:55-62)
//   and `axiom_policy` only writes bias[0..3). SUMMARY: [5..48) of L2 and the bypass
//   [48..96) feed no decision in ANY pipeline — they are "feature neurons for evolution
//   to explore" (commit 9ac2cdd) that never gained a consumer. Full evidence:
//   docs/PLANO_EVOLUCAO.md "P1.8" plus the L2 investigation of 2026-08-14 (blind lines ≈
//   read lines, carried along by elitism's genetic hitchhiking).
//
// # Memory layout (SoA — Structure of Arrays)
//
// Why SoA and not AoS?
// With 60 beings and 192 × 96 = 18 432 L1 weights each, all of L1 is 60 × 18 432 × 4 B ≈
// 4.4 MB. (The earlier comment quoted "223 × 96 = 21 408 weights, ~1.3 MB", which matched
// neither the pre-migration wave nor the current one.) The L1 matmul has thread `j`
// (output neuron j) read the whole row weights[j*WAVE .. (j+1)*WAVE] out of its own
// being's memory. With SoA (all of being 0 → all of being 1 → …) the 96 threads of one
// workgroup read distinct rows of the SAME being — a sequential burst read from VRAM.
// With AoS (being0_w0, being1_w0, …, being0_w1, being1_w1, …) the accesses would be
// interleaved across beings, with no coalescing. SoA is the right choice.
//
// # Recurrent state (CfC state) resident on the GPU
//
// `buf_cfc` (binding 6) survives between ticks. The kernel reads the state, applies the
// CfC equation and writes it back. The CPU only touches this buffer:
//   - on spawn: writes zeros through `reset_cfc(being_idx)`
//   - on death: may download it for a log (a rare operation that does not block a tick)
//
// # Workgroup memory
//
// One workgroup per being, 128 threads. LDS in use:
//   w_kan[223]  i32, post-KAN          : 223 × 4 =  892 B
//   w_cfc[96]   i32, post-CfC          :  96 × 4 =  384 B
//   w_l1r[96]   f32, L1 accumulators   :  96 × 4 =  384 B
//   w_l1a[96]   i32, quantised L1      :  96 × 4 =  384 B
//   w_l2r[48]   f32, l2_raw pre-gain   :  48 × 4 =  192 B  (P1.5, norm_l2_by_std)
//   Total: ~2.2 KB per workgroup. RDNA2 guarantees ≥ 32 KB of LDS per workgroup.
//
// # Expected GPU × CPU divergence
//
// 1. tanh/exp: WGSL is IEEE 754-like, ≤ 1 ULP for |x| < 4. Saturation at ±1 happens on
//    both sides before any quantisation. Difference ≤ 1 LSB after the i8 step.
//
// 2. L1/L2 matmul: sequential f32 accumulation inside each thread, in the same order as
//    the CPU (in_idx from 0 to N). Expected divergence < 1e-5 per neuron.
//
// 3. CfC deterministic noise: a u32 hash with wrapping — identical to the CPU (Rust
//    `wrapping_mul` == a WGSL u32 operation, which wraps by default).
//    NOTE: `denoise_counter` is stored alongside the state in buf_cfc.
//
// 4. KAN axiom 1 (|x| < 0.001 → 0): f32 without FTZ — identical to the CPU.
//
// # What stayed on the CPU, and why
//
// - Trinity (Gate + Reflex + Sentinel): it reads a SafetyState with optional Bevy fields
//   (opt_body) and touches global ablation state (`crate::neuro::ablada()`).
//   Cost: ~2% of the total. Heavy, divergent branching.
//
// - axiom_policy: 25+ branches over per-being boolean flags. Highly divergent on a GPU —
//   the lanes serialise. rayon parallelises it well on the CPU.
//
// - Myelination (top-K sort by resonance_coherence): a rare event; sorting N=96 would be
//   possible but is not worth the code.
//
// - Predictive coding, mirror neurons, metabolism: they depend on Bevy state that was
//   never tensorised. Cost < 0.5% of the total.
//
// - SlowBrain (CfC-64 → Dense → Dense, runs once every 8 ticks): ~1.7% of a tick, not
//   worth porting the DenseLayer state separately.

// ── CONSTANTS ─────────────────────────────────────────────────────────────────

// Sensory-wave size (config.rs:309, SENSORY_WAVE_SIZE, computed — currently 192).
const WAVE:      u32 = 192u;

// u32 words of the packed wave (4 i8 channels per u32). This MUST match WAVE_U32S in the
// perception shaders (perception*.wgsl = 48). It was hardcoded to 56u here, from the days
// when the wave had 223 channels; when the wave shrank to 192 this site was missed, so the
// brain read perception with stride 56 while it was written with stride 48 — every being
// read its neighbour's data. Measured and fixed in that session.
const WAVE_U32S: u32 = 48u;   // ceil(192/4)

// L1: BitLinear WAVE→96 (config.rs: L1_NEURONS = 96)
const L1_OUT:    u32 = 96u;

// L2: FloatLinear 96→48 (config.rs: L2_NEURONS = 48)
const L2_OUT:    u32 = 48u;

// KAN: 16 knots per channel (kan.rs: spline_coeffs [f32; 16])
const KAN_KNOTS: u32 = 16u;

// CfC parameters.
//
// τ IS NO LONGER A CONSTANT. This is where `CFC_DECAY_RATE = 0.5` and
// `CFC_INPUT_GAIN = 1.0` used to live: one τ of 2.0 s identical across all 96 channels,
// walls, curvature and speed alike. The rate now comes from `buf_tau` — one value per
// channel per being, inherited from the genome (mirroring `cfc.rs: decay_rates`).
//
// `CFC_DC_GAIN` is the steady-state gain, the SAME for every τ. The old pair gave
// input_gain/decay = 1.0/0.5 = 2.0, so 2.0 preserves the signal's scale. Without that
// normalisation a fast channel would come out ~80× weaker and be discarded for being
// mute rather than for being fast (see `cfc.rs: dc_gain`).
const CFC_DC_GAIN:     f32 = 2.0;
const CFC_CLAMP:       f32 = 2.0;
const CFC_DENOISE_MAG: f32 = 0.10;  // ~14% of the channels perturbed per tick

// Adaptive fatigue parameters (spike-frequency adaptation).
//
// Mechanism: a neuron that fires hard for a long time reduces its own gain. It is what a
// real brain does to avoid a fixed point and irreversible saturation.
//
// Equation per channel i, every tick:
//   fatigue[i] += FATIGUE_ALPHA * (|state[i]| - fatigue[i]) * dt
//   x_effective[i] = sign(x[i]) * max(0, |x[i]| - FATIGUE_BETA * fatigue[i])
//
// FATIGUE_ALPHA = 1.0: a low-pass filter with τ ≈ 1 s over |state|.
//   With dt=1/64, alpha*dt ≈ 0.0156 → it rises to 63% of the target in ~64 ticks (1 s).
//   With the state saturated at 2.0, fatigue reaches ~1.8 in ~2 s (128 ticks).
//   Faster than the 0.3 version: it guarantees fatigue rises before the fixed point
//   consolidates (which happened in ~1000-1200 ticks with alpha=0.3).
//
// FATIGUE_BETA = 0.4: a threshold applied to the MAGNITUDE of the input.
//   With fatigue=1.8 and beta=0.4 the threshold is 0.72: the channel LOSES gain while it
//   is saturated without going mute, and recovers as soon as the fatigue eases. The fixed
//   point for a sustained maximum input is s = 2/(1+2β) = 1.111 (output 102/127) — the
//   channel keeps informing, which is the contract tested in `cfc.rs`. With β=0.8 it
//   would be 0.769 (output 82).
//   It does NOT invert the sign: the previous version subtracted directly, and a
//   sustained positive input came out NEGATIVE, which breaks the contracts tested in
//   cfc.rs. Normal driving: the being oscillates between throttle and steering → |state|
//   varies → fat varies → fatigue never climbs to its maximum (only sustained firing,
//   i.e. a fixed point, does that).
const FATIGUE_ALPHA: f32 = 1.0;
const FATIGUE_BETA:  f32 = 0.4;  // see cfc.rs: 2/(1+2β) preserves the sustained response

// Adaptive fatigue on the OUTPUT (output fatigue, slots [96..192] of buf_fatigue per being)
//
// Diagnosed problem: intent[0] (throttle) saturated at +1.0 from tick 10 onwards and
// never let go — 46 249 alerts of |v|>0.9. The CfC (input) fatigue breaks a fixed point
// in the recurrent state, but it does not reach the L2/bypass output directly.
//
// Mechanism: the same low-pass filter, applied to the final_intent output.
//   out_fat[i] += OUT_ALPHA * (|intent_raw[i]| - out_fat[i]) * dt
//   intent_final[i] = tanh(sign(raw) * max(0, |raw| - params.out_beta * out_fat[i]))
//
// OUT_ALPHA = 4.0: τ ≈ 0.25 s (it rises to 63% of the target in ~16 ticks).
//   A faster response: the throttle locks in 10 ticks, so the fatigue has to act inside
//   16 ticks to break the output fixed point.
//   (v2: 2.0→4.0 for faster convergence — at 2.0 it took 32 ticks to reach 63% of the
//    target, far too slow when l2_raw is large and the marginal subtraction was not
//    enough.)
//
// OUT_BETA = 0.85: shrink the magnitude of raw by 0.85 × fat before the tanh.
//   Symmetric by construction — `out_fat` tracks |raw|, so subtracting with the sign
//   amplified the negative side and attenuated the positive one (a downward bias across
//   the whole population). Shrinking the magnitude treats both sides alike.
//   In steady state (out_fat → |l2_raw|), intent → tanh(|l2_raw| × (1-0.85))
//                                                  = tanh(|l2_raw| × 0.15)
//   With |l2_raw| = 3.8 (typical when saturated): tanh(0.57) ≈ 0.54 → below 0.90 ✓
//   With |l2_raw| = 4.5 (extreme case):           tanh(0.675) ≈ 0.59 → below 0.90 ✓
//   With a varying signal (l2_raw oscillating), out_fat does not converge to
//   |l2_raw_max| and settles at an intermediate level, so a legitimate signal is not
//   penalised excessively.
//   (v2: 0.6→0.85 — with 0.6, steady state was tanh(l2_raw*0.4), which for l2_raw=3.8
//    still gave tanh(1.52) ≈ 0.91, marginally above the 0.90 threshold.)
const OUT_ALPHA: f32 = 4.0;
// OUT_BETA left this file: it now comes from `params.out_beta`, per being.

// ── STRUCTS ───────────────────────────────────────────────────────────────────

struct Params {
    n_beings:          u32,
    dt:                f32,
    // When 1: ignore buf_wave (binding 1) and unpack the sensory wave straight out of
    // buf_percept (binding 8), which holds the output of perception.wgsl (48 u32 per
    // being, i8 packed little-endian). This removes the perception→CPU readback and the
    // CPU→GPU write_buffer that used to happen. gpu_validate passes 0 here.
    use_percept_direct: u32,
    // Output fatigue strength. Used to be `const OUT_BETA = 0.85`; it became a parameter
    // so it could be measured in a paired test (see `BrainParams::out_beta` on the host).
    out_beta:          f32,
    // P1.5: gain applied to l2_raw and to the L1 bypass before the fatigue/tanh in
    // STAGE 2. Used to be `l2_raw = acc * (2.0/127.0)` and `raw_bp = w_l1r[tid] * 2.0` —
    // two hardcoded `2.0`s. The default of 2.0 (`brain.rs::OUT_GAIN_DEFAULT`) reproduces
    // the historical behaviour exactly; the A/B tests 2.0 vs 1.0 vs 0.5
    // (docs/PLANO_EVOLUCAO.md P1.5) against L2 saturation (48/48 neurons active in 97.4%
    // of the ticks).
    out_gain:          f32,
    // P1.5, second lever (default OFF=0): 1 normalises l2_raw by the standard deviation
    // of the 48 l2_raw values of the SAME being before the gain/tanh — it attacks the
    // SCALE of the L2 input (which mutation can grow across generations) rather than only
    // the output gain. See STAGE 2 below.
    norm_l2_by_std:    u32,
    // P1.7 (docs/PLANO_EVOLUCAO.md) — the control arm of the asynchronous mechanisms, one
    // bit each. This field used to be `_pad0`, dead padding.
    // bit0 = memory (mechanism 1) off: alpha treated as 0.0 in the shader, REGARDLESS of
    //        what the genome carries (the same contract as the slow-brain ablation flag
    //        at main.rs:547: the data may exist and may have mutated, but the bit is what
    //        decides whether it is READ).
    // bit1 = period (mechanism 2) off: period treated as 1 (recompute every tick, as
    //        today).
    // bit2 = threshold (mechanism 2) off: threshold treated as 0.0 (a contribution is
    //        never zeroed for being weak).
    // bit3 = heterogeneous activation (mechanism 3) off: the id is treated as 0 (tanh,
    //        today's single non-linearity).
    // 0 (the default) means every mechanism READS the genome normally. A newborn genome
    // (`Genome::random`) has all four fields at their neutral value (0.0/1.0/0.0/0), so
    // even with the mask at 0 the output of generation 0 is identical to the old formula
    // — tested in `gpu::evolution::tests::genoma_novo_tem_mecanismos_p17_neutros`.
    mech_off_mask:     u32,
    world_tick:        u32,
    // D3 (docs/ESCADA_DIFICULDADE.md) — sensor noise: standard deviation of the
    // approximate Gaussian noise added to EVERY sensory-wave channel, EVERY tick, in i8
    // channel units (-127..127). For example sigma=6 ≈ 5% of the one-sided range (127).
    // Applied BEFORE the KAN, in STAGE 0A (see `apply_sensor_noise`).
    // 0.0 (the default) = no noise, behaviour identical to the historical path.
    noise_sigma:       f32,
    // D3 — channel failure (sensor dropout): probability in [0,1] that the channel is
    // ZEROED on this tick ("the sensor did not answer"), drawn per channel per tick
    // through a deterministic hash (the same pattern as the CfC denoise below).
    // 0.0 (the default) = no channel fails, behaviour identical to the historical path.
    dropout_prob:      f32,
    // Lifetime mechanisms: bit 0 plasticity, bit 1 age/energy development,
    // bit 2 deterministic topology density masks. Zero is exactly the old path.
    lifetime_flags:    u32,
    // Padding: keeps the uniform a multiple of 16 bytes (12 fields × 4 = 48, the same
    // convention as the original struct, which was 8 × 4 = 32). Mirrors `BrainParams` on
    // the host.
    ablation_from_tick:u32,
    // bit 0 = ablation enabled; bits 1..31 = observed being + 1 (zero = disabled)
    control_flags:     u32,
    lifetime_reserved0:u32,
    lifetime_reserved1:u32,
    lifetime_reserved2:u32,
}

// ── BINDINGS ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>            params:     Params;

// The sensory wave as i32 (sign-extended from i8 by the host, which keeps bit tricks out
// of the shader). Layout: being * WAVE + channel
@group(0) @binding(1) var<storage, read>      buf_wave:   array<i32>;

// KAN: spline_coeffs[16] per channel per being.
// Layout: being * WAVE * KAN_KNOTS + channel * KAN_KNOTS + knot_idx
@group(0) @binding(2) var<storage, read>      buf_kan:    array<f32>;

// L1 weights (BitLinear WAVE→96, f32 at runtime).
// SoA layout: being * (L1_OUT * WAVE) + output_neuron * WAVE + input_channel
@group(0) @binding(3) var<storage, read>      buf_l1w:    array<f32>;

// L2 weights (FloatLinear 96→48, f32).
// SoA layout: being * (L2_OUT * L1_OUT) + output_neuron * L1_OUT + input_channel
@group(0) @binding(4) var<storage, read>      buf_l2w:    array<f32>;

// Output: final_intent[96] f32 per being.
// Layout: being * 96 + neuron
@group(0) @binding(5) var<storage, read_write> buf_intent: array<f32>;

// CfC state, resident between ticks (96 state floats + 1 u32 counter).
// Layout per being: [state[0], …, state[95], counter_u32_bitcast_f32] = 97 × f32
// Offset per being: being * 97
@group(0) @binding(6) var<storage, read_write> buf_cfc:    array<f32>;

// TEMPORARY DEBUG: exports l1_raw (96 f32) plus the inputs of being 0 (96 + 127 i32).
// Layout: [l1_raw[0..96], input_cfc[0..96], input_kan[96..223]]
// Size: 96*4 + 96*4 + 127*4 = 1276 bytes. The third block is still sized for the
// 223-channel wave; with WAVE=192 only its first 96 entries are ever written (see the
// `j < 127u` guard in STAGE 1, which never binds because only 96 threads reach it).
@group(0) @binding(7) var<storage, read_write> buf_debug:  array<f32>;

// Output of perception.wgsl: WAVE_U32S = 48 u32 per being (192 i8 packed little-endian).
// Layout: buf_percept[being * WAVE_U32S + ch/4], byte (ch % 4), read as i8.
// Used ONLY when params.use_percept_direct == 1 (gpu_loop); ignored by gpu_validate,
// which uses buf_wave (binding 1) with i32 sign-extended by the CPU.
// The type is u32 rather than atomic<u32> because we only READ here (perception.wgsl has
// already finished).
@group(0) @binding(8) var<storage, read>       buf_percept: array<u32>;

// Adaptive fatigue per CfC neuron — slow state, resident between ticks.
// Layout: being * L1_OUT + channel (96 f32 per being, initialised to 0.0).
// It does not keep its own counter: it uses the same counter as buf_cfc.
// It grows when |state[i]| is high, and is subtracted from the CfC input, breaking the
// fixed point.
@group(0) @binding(9) var<storage, read_write> buf_fatigue: array<f32>;

// CfC time constants, one per channel per being — the "τ bank".
// Layout: being * L1_OUT + channel (96 f32 per being). It stores the DECAY RATE in s⁻¹
// (d = 1/τ), not τ, because that is the form that goes straight into the exponential.
// Read-only: τ is inherited from the genome and fixed for the being's life. Modulating τ
// in real time was measured and does not pay (+0.11%); HAVING varied τ is what gets
// selected (40× versus 5.3× in the control).
@group(0) @binding(10) var<storage, read>      buf_tau:     array<f32>;

// ── P1.7 — ASYNCHRONOUS BRAIN (docs/PLANO_EVOLUCAO.md §P1.7) ──────────────────
//
// Three heritable mechanisms, all per-neuron, all born NEUTRAL (identical to today's
// brain) — see `gpu::evolution::Genome` for the fields and `evolve()` for the mutation.
// The owner's critique that motivated this: *"what I am asking for is an ASYNCHRONOUS
// brain; it should work on different dimensions, neurons and parts depending on where it
// is on the track; if you stay on the same tanh forever you are just burning time."* —
// today the 20 uses of tanh in this file are the ONLY non-linearity, and the 96+48
// neurons of L1/L2 fire on EVERY tick (measured, not assumed).
//
// Per-neuron metadata — READ-ONLY, uploaded once per generation (the same pattern as
// `buf_tau`: the data is inherited from the genome and fixed for the being's life).
// Layout per being, NEURON_META_PER_BEING = 480 f32, contiguous blocks:
//   [0..96)    l1_alpha       (mechanism 1 — L1 memory, ∈[0,0.95])
//   [96..192)  l1_period      (mechanism 2 — L1 recompute period, an integer ≥1 stored as
//                              f32, the same disk/genome convention as every other field
//                              — f32 only, see genome_io.rs)
//   [192..288) l1_threshold   (mechanism 2 — L1 contribution threshold, ∈[0,~1))
//   [288..384) l1_activation  (mechanism 3 — L1 activation id: 0=tanh 1=relu 2=linear
//                              3=leaky-relu(0.01), stored as f32)
//   [384..432) l2_alpha       (mechanism 1 — L2 memory, ∈[0,0.95])
//   [432..480) l2_activation  (mechanism 3 — L2 activation id)
const NEURON_META_PER_BEING: u32 = 480u;
// Offsets of the blocks inside each per-being block of `buf_neuron_meta` (see the layout
// above). `l1_alpha` sits at offset 0, implicitly, with no constant of its own.
const NEURON_META_L1_PERIOD_OFF:     u32 = 96u;
const NEURON_META_L1_THRESHOLD_OFF:  u32 = 192u;
const NEURON_META_L1_ACTIVATION_OFF: u32 = 288u;
const NEURON_META_L2_ALPHA_OFF:      u32 = 384u;
const NEURON_META_L2_ACTIVATION_OFF: u32 = 432u;
@group(0) @binding(11) var<storage, read>      buf_neuron_meta: array<f32>;

// Persistent state of the per-neuron EMA memory (mechanism 1) — FOLLOWS THE PATTERN of
// `buf_cfc`: it lives on the GPU between ticks, read_write, is cleared on spawn and
// between generations (`BrainGpu::reset_cfc` now zeroes this buffer alongside
// buf_cfc/buf_fatigue — see the host), and is NEVER downloaded per tick. It holds h[n-1]
// — see the amplitude-gain derivation in the comment on `neuron_mem_gain` below.
// Layout per being, NEURON_MEM_PER_BEING = 144 f32: [0..96) L1 memory, [96..144) L2.
const NEURON_MEM_PER_BEING: u32 = 144u;
@group(0) @binding(12) var<storage, read_write> buf_neuron_mem: array<f32>;

// M4 (REGIOES_CEREBRAIS.md): a multiplicative gate per REGION — 4 f32 per being (blocks
// of 24 L1 neurons: vision / wall / body / associative), written by the CPU from the slow
// brain plus its reserve. 1.0 is neutral (flag off, or the reserve exhausted). Applied to
// the L1 activation before quantisation for L2. The historical bypass reads `w_l1r`
// directly and therefore does not consume this gate.
@group(0) @binding(13) var<storage, read>      buf_gates:  array<f32>;

// Counterfactual output masks. Eight u32 words per being:
// CfC [0..3), L1 [3..6), L2 [6..8). A set bit removes only the emitted value;
// recurrent state and fatigue continue to update under the counterfactual trajectory.
const ABLATION_WORDS_PER_BEING: u32 = 8u;
const ABLATION_CFC_WORD: u32 = 0u;
const ABLATION_L1_WORD:  u32 = 3u;
const ABLATION_L2_WORD:  u32 = 6u;
@group(0) @binding(14) var<storage, read> buf_ablation_mask: array<u32>;

// One selected-being snapshot, stored as raw u32 words so f32 and signed i32 stages can
// share one stable binary layout. Offsets mirror `brain/observation.rs`.
const OBS_L1_MATMUL: u32 = 0u;
const OBS_L1_EFFECTIVE: u32 = 96u;
const OBS_L1_TO_L2: u32 = 192u;
const OBS_L2_MATMUL: u32 = 288u;
const OBS_L2_EFFECTIVE: u32 = 336u;
const OBS_L2_ADAPTED: u32 = 384u;
const OBS_BYPASS_ADAPTED: u32 = 432u;
const OBS_L2_STD: u32 = 480u;
@group(0) @binding(15) var<storage, read_write> buf_observation: array<u32>;

// Optional resident lifetime state. Fixed per-being stride: config[32], feedback[4],
// P[144], eligibility[144], previous latent[48], previous motor[3], runtime summaries.
const LIFE_STRIDE: u32 = 512u;
const LIFE_FEEDBACK: u32 = 32u;
const LIFE_P: u32 = 36u;
const LIFE_E: u32 = 180u;
const LIFE_PREV_LATENT: u32 = 324u;
const LIFE_PREV_MOTOR: u32 = 372u;
const LIFE_RUNTIME: u32 = 375u;
const LIFE_PLASTICITY: u32 = 1u;
const LIFE_DEVELOPMENT: u32 = 2u;
const LIFE_TOPOLOGY: u32 = 4u;
@group(0) @binding(16) var<storage, read_write> buf_lifetime: array<f32>;

// ── WORKGROUP MEMORY ──────────────────────────────────────────────────────────

// w_kan is still sized 223, from the pre-migration wave; only [0, WAVE) is ever used.
var<workgroup> w_kan: array<i32, 223>;  // post-KAN, pre-CfC: every channel
var<workgroup> w_cfc: array<i32, 96>;   // CfC output (tanh(state) × 127, as i32)
var<workgroup> w_l1r: array<f32, 96>;   // raw L1 accumulators (pre-tanh)
var<workgroup> w_l1a: array<i32, 96>;   // L1 quantised as L2 input (i8 × 127)
var<workgroup> w_l2r: array<f32, 48>;   // l2_raw PRE-gain (acc/127, without params.out_gain)
                                          // — used only when params.norm_l2_by_std == 1

// ── KAN filter_pulse (equivalent to KanAxiomFilter::filter_pulse, kan.rs:18-29) ─

// A variant that takes raw_i32 (an integer signal in [-127,127]) and returns the value
// quantised after the KAN, already on the i32 (i.e. i8-equivalent) scale.
//
// NUMERICAL FIX: the previous version computed raw_f32 = raw_i32/127.0, then
// filtered = raw_f32 * coeff, then w_kan = i32(filtered * 127.0). That amounts to
// raw_i32 / 127.0 * coeff * 127.0. The /127 *127 round trip introduced 1 ULP of
// difference against the CPU, causing a boundary crossing for some values (for instance
// -18/127*1.0*127 is not exactly -18 in f32 on the GPU). Avoid the round trip: use
// raw_i32 * coeff directly, without dividing and multiplying by 127. The result is
// clamp(raw_i32 * coeff, -127, 127).
fn kan_filter_int(raw_i32: i32, knots_base: u32) -> i32 {
    let raw_f32: f32 = f32(raw_i32);

    // Axiom 1: a zero signal carries no direction (kan.rs:20-21).
    // On the i32 scale: |raw_i32| = 0 (identically zero) → 0.
    // The original threshold was |raw_f32/127| < 0.001 → |raw_f32| < 0.127 → |raw_i32| < 1
    // in integers. Since raw_i32 is an integer, only raw_i32 == 0 strictly satisfies
    // |raw_i32| < 0.127.
    // NOTE: the value 1 in i8 gives raw_f32_original = 1/127 ≈ 0.00787, which is > 0.001,
    // so it is not zeroed. Hence the correct threshold in i32 is: raw_i32 == 0.
    if raw_i32 == 0 { return 0; }

    // Spline index: mapped from raw_f32_normalized ∈ [-1,1] onto [0,15].
    // raw_f32_normalized = raw_i32 / 127.0 (but the index can be computed without dividing)
    // idx = (raw_i32/127.0 + 1.0) * 7.5 = (raw_i32 + 127.0) * 7.5 / 127.0
    //     = (f32(raw_i32) + 127.0) * (7.5 / 127.0)
    let idx_f: f32 = (raw_f32 + 127.0) * (7.5 / 127.0);
    let idx: u32   = u32(clamp(idx_f, 0.0, 15.0));
    let c: f32     = buf_kan[knots_base + idx];

    // Result without the round trip: raw_i32 * coeff (on the i8 / 127 scale).
    // Equivalent to: (raw_i32/127 * coeff) * 127 = raw_i32 * coeff
    return i32(clamp(raw_f32 * c, -127.0, 127.0));
}

// ── P1.7: heterogeneous per-neuron activation (mechanism 3) ───────────────────
//
// ids: 0=tanh (the default, today) · 1=ReLU · 2=linear · 3=leaky-ReLU(0.01).
// Same formulas as `math::activations` (src/math/activations.rs) — this does NOT
// reimplement new logic, it only mirrors what already lives there (task rule: "the
// functions ALREADY EXIST, use them from there"). The clamp to the i8 range happens in
// the CALLER (`clamp(activated * 127.0, -127.0, 127.0)`), exactly as it already did for
// tanh — linear and ReLU, which lack tanh's natural saturation, depend entirely on that
// external clamp not to overflow the quantisation.
//
// COST OF WARP DIVERGENCE (RDNA2, wave32): the 32 lanes of a wave execute in LOCKSTEP
// (SIMD). If the neurons mapped onto the lanes of the SAME wave carry different ids, the
// hardware has to walk every branch that ANY lane of the wave took — each lane only
// stores the result of its own case (predication), but the time paid is that of executing
// every distinct case present in the wave, not just its own. With 96 L1 neurons = 3 waves
// of 32 (the RDNA2 wave size, see the project's CLAUDE.md) and a LOW activation mutation
// rate (see `evolve()` in `gpu/evolution.rs`), the fraction of heterogeneous waves grows
// slowly across generations — the cost is real (not free) but small: the 4 cases are pure
// ALU (no divergent memory access), a handful of instructions each.
fn apply_activation(id: u32, x: f32) -> f32 {
    if id == 1u { return max(x, 0.0); }                     // ReLU
    if id == 2u { return x; }                                // linear (saturates in the caller's clamp)
    if id == 3u { return select(x, x * 0.01, x < 0.0); }      // leaky-ReLU(0.01)
    return tanh(x);                                           // 0 = tanh (the default, identical to today)
}

// ── P1.7: per-neuron memory (mechanism 1) ─────────────────────────────────────
//
// h[n] = h[n-1]·alpha + x[n]·(1-alpha) — a convex combination, the coefficients sum to 1.
// For a CONSTANT INPUT this already has DC gain 1 at ANY alpha (solve h* = h*·alpha +
// x·(1-alpha) ⇒ h* = x), unlike the CfC (`buf_cfc` above), which needs the
// `CFC_DC_GAIN·(1-decay)` factor because there the "gain" and the "decay" are independent
// free parameters. There is no such artefact here — but there is ANOTHER one: the
// AMPLITUDE (variance) of a signal that FLUCTUATES tick to tick (the real case for the
// sensory channels — wall, curvature and speed are never constant) IS attenuated by a
// low-pass filter, even with the DC gain exactly at 1. Modelling x[n] as approximately
// white noise of variance σx² (the filter's impulse response being
// h[n] = (1-alpha)·Σ_{k=0}^{n} alpha^k·x[n-k]):
//
//   Var(h) = (1-alpha)²·σx²·Σ alpha^{2k} = (1-alpha)²/(1-alpha²)·σx²
//          = (1-alpha)/(1+alpha)·σx²
//
// At alpha=0.95, Var(h)/Var(x) ≈ 0.026 — the neuron would look 97% weaker (MUTE) and be
// discarded for being weak, not for being slow (GAP 11 of the project's CLAUDE.md). The
// gain that RESTORES Var(output) = Var(x) is the inverse square root of that factor:
//
//   gain(alpha) = sqrt((1+alpha)/(1-alpha))     [gain(0)=1 — nothing changes at alpha=0]
//
// At alpha=0.95, gain ≈ 6.24 — large, but finite because alpha is clamped at 0.95 in
// `evolve()` (alpha≥1 diverges; the stability condition of the difference equation puts
// the filter's pole at |alpha|<1). `max(1.0-alpha, 1e-3)` is only a second numerical
// guard (the clamp in `evolve()` already forbids alpha=1; this is defence in depth
// against a corrupted or legacy genome out of range).
const NEURON_MEM_GAIN_EPS: f32 = 1e-3;
fn neuron_mem_gain(alpha: f32) -> f32 {
    return sqrt((1.0 + alpha) / max(1.0 - alpha, NEURON_MEM_GAIN_EPS));
}

// ── D3 (docs/ESCADA_DIFICULDADE.md): sensor noise + channel failure ───────────
//
// Applied in STAGE 0A, BEFORE the KAN (`kan_filter_int`) — it simulates a noisy or
// unstable sensor at the point where the wave is read, before any neural processing (the
// KAN filter and the CfC receive an already degraded signal, just like a real sensor).
//
// # Deterministic hash — the same pattern as the CfC denoise above
// (`h = counter*2654435761u + i`, Knuth's hash, wrapping u32 — Rust `wrapping_mul` /
// `wrapping_add` == WGSL u32, which wraps by default).
//
// Extended with `being * 40503u`: the CfC denoise uses only `counter*prime + channel`
// because there the counter is already PER BEING (read from buf_cfc[being*97+96]) — but
// here we need the SAME counter for all of a being's channels inside the SAME dispatch,
// so we read the counter ONCE (`sensor_counter`, see STAGE 0A) — and it is identical
// across beings in the same generation (they all increment once per tick in lockstep).
// Without the `being` term, two different beings would see the SAME noise on the SAME
// channel on the SAME tick — a degeneracy the CfC denoise does not have (its counter is
// already read from THAT being's buf_cfc). `40503u` is just another large odd multiplier
// (it does not have to be Knuth's prime) to mix `being` in without colliding with the
// channel term.
//
// It depends on (tick, through sensor_counter; being; channel) — deterministic replay is
// preserved: the same seed and the same sequence of ticks give the same noise.
//
// # Approximate Gaussian
//
// The sum of 3 independent uniforms on (0,1) — 3 chained hashes, each taking bits [8..31]
// of the hash, the same extraction `f32(h>>8u)/16777216.0` the CfC denoise already uses —
// minus the mean 1.5, scaled by 2:
//
//   Var(U(0,1)) = 1/12  ⇒  Var(sum of 3) = 3/12 = 0.25  ⇒  std dev = 0.5
//   (sum - 1.5) has mean 0 and std dev 0.5 → ×2 normalises it to a std dev of ≈ 1.0
//
// A CLT (Central Limit Theorem) approximation with few terms — it is NOT an exact
// Gaussian (that would need ~12 terms, the classic Irwin-Hall trick where the sum of 12
// U(0,1) already has a std dev of exactly 1 without scaling), but 3 terms already break
// the rectangular shape of a single uniform and give a bell-shaped distribution, which is
// enough to simulate sensor noise. Documented, not hidden: this is NOT Box-Muller, which
// would cost a log/sin/cos per channel per tick × 192 channels × N beings — far too
// expensive for the fidelity it would buy.
//
// `noise_sigma` is in i8 CHANNEL UNITS (-127..127): sigma=6 ≈ 5% of the one-sided range
// (127 is the amplitude, not the 254 end to end).
//
// # Dropout (channel failure)
//
// A separate hash (`counter+7u`, with the same being/channel mixing) so the failure draw
// does not correlate with the noise draw above — a channel can drop even on a
// tick/being/channel where the noise happened to land near zero, and vice versa. When the
// sensor "fails", the channel comes out ZEROED for the rest of the pipeline (the KAN sees
// 0, and axiom 1 of kan_filter_int already treats raw==0 as "no direction").
fn apply_sensor_noise(raw_i32: i32, being: u32, ch: u32, counter: u32) -> i32 {
    var v: i32 = raw_i32;

    if params.noise_sigma > 0.0 {
        let h0: u32 = counter * 2654435761u + being * 40503u + ch;  // Knuth hash + being + channel
        let h1: u32 = h0 * 2654435761u + 1u;                         // 2nd chained stage
        let h2: u32 = h1 * 2654435761u + 1u;                         // 3rd chained stage
        let u0: f32 = f32(h0 >> 8u) / 16777216.0;                    // uniform [0,1)
        let u1: f32 = f32(h1 >> 8u) / 16777216.0;
        let u2: f32 = f32(h2 >> 8u) / 16777216.0;
        let gauss_approx: f32 = (u0 + u1 + u2 - 1.5) * 2.0;           // mean 0, std dev ≈ 1.0
        let noisy: f32 = f32(v) + gauss_approx * params.noise_sigma;
        v = i32(clamp(noisy, -127.0, 127.0));
    }

    if params.dropout_prob > 0.0 {
        let hd: u32 = (counter + 7u) * 2654435761u + being * 40503u + ch;  // hash independent of the noise one
        let ud: f32 = f32(hd >> 8u) / 16777216.0;
        if ud < params.dropout_prob {
            v = 0;  // the sensor failed: the channel is zeroed on this tick
        }
    }

    return v;
}

fn output_is_ablated(being: u32, stage_word: u32, neuron: u32) -> bool {
    if (params.control_flags & 1u) == 0u || params.world_tick < params.ablation_from_tick {
        return false;
    }
    let base = being * ABLATION_WORDS_PER_BEING + stage_word;
    let word = buf_ablation_mask[base + neuron / 32u];
    return (word & (1u << (neuron % 32u))) != 0u;
}

fn observes(being: u32) -> bool {
    return (params.control_flags >> 1u) == being + 1u;
}

fn life_base(being: u32) -> u32 {
    return being * LIFE_STRIDE;
}

fn life_gene(being: u32, slot: u32) -> f32 {
    return buf_lifetime[life_base(being) + slot];
}

fn life_feedback(being: u32, slot: u32) -> f32 {
    return buf_lifetime[life_base(being) + LIFE_FEEDBACK + slot];
}

// Age and energy are capacity masks.  They gate the values entering the existing
// layers; all 96/48 lanes still execute on the GPU, so this is a topology mask and
// not a claim of compute savings.
fn life_neuron_active(being: u32, layer: u32, neuron: u32) -> bool {
    if (params.lifetime_flags & (LIFE_DEVELOPMENT | LIFE_TOPOLOGY)) == 0u { return true; }
    var capacity = 1.0;
    if (params.lifetime_flags & LIFE_DEVELOPMENT) != 0u {
        let age = clamp(buf_lifetime[life_base(being) + LIFE_RUNTIME], 0.0, 1.0);
        let energy = clamp(life_feedback(being, 1u), 0.0, 1.0);
        let start = select(life_gene(being, 8u), life_gene(being, 4u), layer == 1u);
        let end = select(life_gene(being, 9u), life_gene(being, 5u), layer == 1u);
        let energy_min = select(life_gene(being, 10u), life_gene(being, 6u), layer == 1u);
        let progress = clamp((age - start) / max(end - start, 1e-3), 0.0, 1.0);
        let energy_progress = clamp((energy - energy_min) / max(1.0 - energy_min, 1e-3), 0.0, 1.0);
        capacity = (0.25 + 0.75 * progress)
            * (0.25 + 0.75 * pow(energy_progress, clamp(life_gene(being, 15u), 0.25, 4.0)));
    }
    if (params.lifetime_flags & LIFE_TOPOLOGY) != 0u {
        let density = select(life_gene(being, 11u), life_gene(being, 7u), layer == 1u);
        capacity = capacity * (0.25 + 0.75 * density);
    }
    if layer == 2u && neuron < 3u { return true; }
    // Preserve a scaffold in EVERY L1 sensory region, rather than keeping only
    // the first (visual) region and depriving a juvenile of body/wall signals.
    let rank = select(neuron, neuron % 24u, layer == 1u);
    let width = select(L2_OUT, 24u, layer == 1u);
    return rank < u32(ceil(f32(width) * capacity));
}

// Stable integer hash of an edge identity.  A density of 1.0 is an exact fast path
// and preserves every historical accumulation bit-for-bit when the feature is off.
fn life_edge_enabled(being: u32, layer: u32, output: u32, input: u32) -> bool {
    if (params.lifetime_flags & LIFE_TOPOLOGY) == 0u {
        return true;
    }
    let density = select(life_gene(being, 11u), life_gene(being, 7u), layer == 1u);
    if density >= 1.0 { return true; }
    if density <= 0.0 { return false; }
    // Being is intentionally absent: identical lifetime genes have identical
    // topology regardless of population slot.
    var h = layer * 0x85ebca6bu + output * 0xc2b2ae35u + input;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h & 0xffffu) / 65535.0 < density;
}

fn life_elapsed(being: u32) -> f32 {
    let base = life_base(being);
    let initialized = buf_lifetime[base + LIFE_RUNTIME + 5u] > 0.0;
    let previous = u32(max(buf_lifetime[base + LIFE_RUNTIME + 1u], 0.0));
    let delta = select(1u, max(params.world_tick - previous, 1u), initialized);
    return max(params.dt * f32(delta), 0.0);
}

fn life_step_elapsed(being: u32) -> f32 {
    return max(buf_lifetime[life_base(being) + LIFE_RUNTIME + 8u], 0.0);
}

// ── MAIN KERNEL ───────────────────────────────────────────────────────────────

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being = wgid.x;    // one workgroup per being
    let tid = lid.x;       // local thread [0, 127]

    // More workgroups than living beings: the standard guard (the dispatch is not rounded).
    if being >= params.n_beings { return; }

    // `world_tick` advances even when the host skips a brain dispatch.  Consume the
    // elapsed tick span here so age and eligibility use physical time rather than the
    // number of neural dispatches.  The runtime slots are touched only when a lifetime
    // flag is active, preserving the baseline buffer's untouched state.
    if params.lifetime_flags != 0u && tid == 0u {
        let base = life_base(being);
        let elapsed = life_elapsed(being);
        buf_lifetime[base + LIFE_RUNTIME + 8u] = elapsed;
        let previous_age = clamp(buf_lifetime[base + LIFE_RUNTIME], 0.0, 1.0);
        if (params.lifetime_flags & LIFE_DEVELOPMENT) != 0u {
            let rate = clamp(life_gene(being, 14u), 0.0, 4.0);
            buf_lifetime[base + LIFE_RUNTIME] = clamp(previous_age + elapsed * rate * clamp(life_feedback(being, 2u), 0.0, 1.0), 0.0, 1.0);
        }
        let reward = life_feedback(being, 0u);
        let energy = clamp(life_feedback(being, 1u), 0.0, 1.0);
        let alive = clamp(life_feedback(being, 2u), 0.0, 1.0);
        let stat_mix = clamp(1.0 - exp(-elapsed), 0.0, 1.0);
        buf_lifetime[base + LIFE_RUNTIME + 2u] =
            buf_lifetime[base + LIFE_RUNTIME + 2u] + (reward - buf_lifetime[base + LIFE_RUNTIME + 2u]) * stat_mix;
        buf_lifetime[base + LIFE_RUNTIME + 3u] =
            buf_lifetime[base + LIFE_RUNTIME + 3u] + (energy - buf_lifetime[base + LIFE_RUNTIME + 3u]) * stat_mix;
        buf_lifetime[base + LIFE_RUNTIME + 4u] =
            buf_lifetime[base + LIFE_RUNTIME + 4u] + elapsed * alive;
        buf_lifetime[base + LIFE_RUNTIME + 5u] =
            buf_lifetime[base + LIFE_RUNTIME + 5u] + 1.0;
        buf_lifetime[base + LIFE_RUNTIME + 1u] = f32(params.world_tick);
    }
    storageBarrier();
    workgroupBarrier();

    let wave_base: u32 = being * WAVE;
    let kan_base:  u32 = being * WAVE * KAN_KNOTS;

    // D3 (docs/ESCADA_DIFICULDADE.md) — the being's counter, the SAME one the CfC denoise
    // uses (buf_cfc[being*97+96]), read HERE (STAGE 0A) — that is, BEFORE the increment
    // STAGE 0B performs further down. Both passes of this stage use this SAME value for
    // every channel: a deterministic hash over (tick, being, channel) through
    // `apply_sensor_noise`; see that function's documentation.
    let sensor_counter: u32 = bitcast<u32>(buf_cfc[being * 97u + 96u]);

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 0A: per-channel KAN (SensoryPreFilter::apply, kan.rs:72-79)
    //
    // WAVE = 192 needs 2 passes of 128 threads.
    //   Pass 1: tid ∈ [0,127] → channel tid       (all 128 threads active)
    //   Pass 2: tid ∈ [0, 63] → channel tid+128   (channels 128..191)
    // The writes into w_kan land on disjoint positions — no conflict.
    //
    // WHERE THE WAVE COMES FROM: two paths, chosen by params.use_percept_direct.
    //   0 = buf_wave (binding 1): i32 sign-extended, uploaded by the CPU through
    //       write_buffer. Used by gpu_validate, which builds the wave on the CPU and
    //       hands it to the GPU.
    //   1 = buf_percept (binding 8): WAVE_U32S = 48 u32 per being, the output of
    //       perception.wgsl.
    //       Layout: byte ch = (buf_percept[being*WAVE_U32S + ch/4] >> ((ch%4)*8)) & 0xFF,
    //       read as two's-complement i8.
    //       Used by gpu_loop: perception → brain INSIDE VRAM, with zero readback.
    // ─────────────────────────────────────────────────────────────────────────

    // Unpacks channel `ch` of the sensory wave according to the read mode. Inlined into
    // both passes below to avoid the function-call overhead.

    // Pass 1: channels 0..127
    {
        let ch: u32 = tid;
        // ch < 128 always holds here; WAVE=192 > 128, so no guard is needed.
        var raw_i32: i32;
        if params.use_percept_direct == 1u {
            // Unpack a two's-complement i8 from buf_percept (perception.wgsl's output).
            // WAVE_U32S u32 per being: byte ch lives in word (ch/4), at shift ((ch%4)*8).
            let percept_base: u32 = being * WAVE_U32S;
            let word: u32   = buf_percept[percept_base + ch / 4u];
            let byte_u: u32 = (word >> ((ch % 4u) * 8u)) & 0xFFu;
            // Sign extension u8 → two's-complement i32: if bit 7 is set (byte >= 128),
            // the value is byte - 256.
            raw_i32 = select(i32(byte_u), i32(byte_u) - 256, byte_u >= 128u);
        } else {
            raw_i32 = buf_wave[wave_base + ch];
        }
        // D3: sensor noise + channel failure, BEFORE the KAN. A no-op when
        // noise_sigma==0 and dropout_prob==0 (the historical behaviour).
        raw_i32 = apply_sensor_noise(raw_i32, being, ch, sensor_counter);
        w_kan[ch] = kan_filter_int(raw_i32, kan_base + ch * KAN_KNOTS);
    }
    // Pass 2: channels 128..191 (64 channels, so threads 64..127 do nothing)
    {
        let ch: u32 = tid + 128u;
        if ch < WAVE {
            var raw_i32: i32;
            if params.use_percept_direct == 1u {
                let percept_base: u32 = being * WAVE_U32S;
                let word: u32   = buf_percept[percept_base + ch / 4u];
                let byte_u: u32 = (word >> ((ch % 4u) * 8u)) & 0xFFu;
                raw_i32 = select(i32(byte_u), i32(byte_u) - 256, byte_u >= 128u);
            } else {
                raw_i32 = buf_wave[wave_base + ch];
            }
            // D3: same treatment as Pass 1 — see the comment above.
            raw_i32 = apply_sensor_noise(raw_i32, being, ch, sensor_counter);
            w_kan[ch] = kan_filter_int(raw_i32, kan_base + ch * KAN_KNOTS);
        }
    }

    workgroupBarrier();

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 0B: CfC pre-processor (CfCCell::update, cfc.rs:59-86)
    //
    // Updates state[0..96] and writes filtered_retina into w_cfc[0..96].
    // Only 96 threads (tid < 96) work — the other 32 idle.
    //
    // State layout in buf_cfc (per being):
    //   [state[0], …, state[95], counter_u32_bitcast_f32] = 97 × f32
    //   offset per being: being * 97
    //
    // Counter: incremented BEFORE the channel loop (cfc.rs:68: wrapping_add(1)). Every
    // thread reads the same counter_old and computes an identical counter_new. Only
    // tid==0 writes counter_new back (a single write, no race).
    // ─────────────────────────────────────────────────────────────────────────

    if tid < L1_OUT {  // L1_OUT = 96 = the CfC width
        let i: u32        = tid;
        let cfc_base: u32 = being * 97u;

        // Current state of channel i
        var s: f32 = buf_cfc[cfc_base + i];

        // Adaptive fatigue: read the slow state from buf_fatigue.
        // Layout: buf_fatigue holds 192 slots per being (FATIGUE_PER_BEING = L1_OUT * 2).
        //   slots [0..96]   per being = CfC INPUT fatigue (this block)
        //   slots [96..192] per being = OUTPUT fatigue (L2 + bypass, the STAGE 2 block)
        //
        // CRITICAL: use being * (L1_OUT * 2u) — NOT being * L1_OUT.
        //   being * L1_OUT (= being * 96) makes being N>0 read the previous being's output
        //   fatigue slots, which cross-contaminates and produced a drift > 1.0 in 100
        //   ticks. The factor 2u guarantees a stride of 192 slots per being, the real
        //   block size.
        let fat_base: u32 = being * (L1_OUT * 2u);
        var fat: f32 = buf_fatigue[fat_base + i];

        // Counter: read by everyone, incremented by one (cfc.rs:68)
        let counter_old: u32  = bitcast<u32>(buf_cfc[cfc_base + 96u]);
        let counter_new: u32  = counter_old + 1u;  // u32 wraps by default in WGSL

        let dt: f32     = params.dt;
        // PER-CHANNEL τ: the decay rate inherited from the genome, in s⁻¹.
        let d: f32       = buf_tau[being * L1_OUT + i];
        let decay: f32   = exp(-d * dt);
        // Normalised DC gain — reuses the `decay` already computed, with no extra exp.
        // Steady state: s* = x·G·(1-decay)/(1-decay) = x·G, for any τ.
        let gain_dt: f32 = CFC_DC_GAIN * (1.0 - decay);

        // x = the channel's normalised input, with fatigue applied as a SOFT THRESHOLD:
        //   x_effective = sign(x) * max(0, |x| - beta * fat)
        // This mirrors `cfc.rs` exactly. The earlier form (`x - beta*fat`, a raw
        // subtraction) INVERTED the sign: with fat ≈ 1.8 and beta = 0.8 it removes 1.44
        // from an x ∈ [-1,1], so a sustained positive input came out negative, and with no
        // input at all the state was pushed away from zero. The tests
        // `update_stabilizes_on_constant_input` and `state_decays_to_zero_without_input` in
        // cfc.rs cover both cases.
        let x_raw: f32   = f32(w_kan[i]) / 127.0;
        let threshold: f32  = FATIGUE_BETA * fat;
        let x: f32       = sign(x_raw) * max(0.0, abs(x_raw) - threshold);

        // CfC equation (cfc.rs:72): state = state * decay + x * gain_dt
        s = s * decay + x * gain_dt;

        // Deterministic hash noise (cfc.rs:74-80).
        // ~14% of the channels are hit per tick (the condition h % 7 == 0).
        // This is not wrapped in an `if CFC_DENOISE_MAG > 0.0` block because WGSL
        // evaluates constants at compile time — the compiler eliminates the branch.
        let h: u32 = counter_new * 2654435761u + i;  // Knuth hash (cfc.rs:75)
        if (h % 7u) == 0u {
            // Noise in [-CFC_DENOISE_MAG, +CFC_DENOISE_MAG] (cfc.rs:77-78)
            let noise: f32 = (f32(h >> 8u) / 16777216.0 - 0.5) * 2.0 * CFC_DENOISE_MAG;
            s = s + noise;
        }

        // Clamp (cfc.rs:82)
        s = clamp(s, -CFC_CLAMP, CFC_CLAMP);

        // Update the fatigue: a low-pass filter over |state| (spike-frequency adaptation).
        // fat += alpha * (|s| - fat) * dt
        // With alpha=0.3 and dt=1/64, fat grows ~0.0047 per tick while |s|=2.0, and
        // reaches 90% of its saturation value in about τ = 1/alpha ≈ 3.3 s ≈ 210 ticks.
        fat = fat + FATIGUE_ALPHA * (abs(s) - fat) * dt;
        buf_fatigue[fat_base + i] = fat;

        // Store the updated state (resident — read again on the next tick)
        buf_cfc[cfc_base + i] = s;

        // filtered = tanh(state) → i8 (cfc.rs:84-85)
        let cfc_emission = i32(clamp(tanh(s) * 127.0, -127.0, 127.0));
        w_cfc[i] = select(cfc_emission, 0, output_is_ablated(being, ABLATION_CFC_WORD, i));

        // Only tid==0 updates the counter (this avoids a write race)
        if i == 0u {
            buf_cfc[cfc_base + 96u] = bitcast<f32>(counter_new);
        }
    }

    workgroupBarrier();

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 1: BitLinear L1 — WAVE→96 (layers.rs:40-51)
    //
    //   l1_raw[j] = Σ_{i=0}^{WAVE-1} weights[j*WAVE+i] * input[i] / 127
    //
    // Composite input: w_cfc[0..96] (post-CfC) ++ w_kan[96..WAVE] (post-KAN).
    // 96 threads active (tid < 96), one per output neuron. Threads 96..127 idle — fine,
    // on RDNA2 that leaves 3 waves with one of them idle.
    //
    // Accumulation order: i from 0 to WAVE-1, sequential, same as the CPU.
    // Expected matmul divergence: < 1e-5 per neuron (192 f32 FMAs).
    //
    // buf_l1w layout: being*(L1_OUT*WAVE) + j*WAVE + i  (row-major per being)
    // ─────────────────────────────────────────────────────────────────────────

    if tid < L1_OUT {
        let j: u32 = tid;
        let l1_base: u32 = being * (L1_OUT * WAVE) + j * WAVE;
        let l1_active = life_neuron_active(being, 1u, j);
        // FIX: accumulate WITHOUT normalising at each step — multiply by norm ONCE at the
        // end, identical to the CPU (layers.rs:47-49):
        //   acc += w[i] * input[i]  (no norm)
        //   output[j] = acc * norm_factor  (norm at the end)
        // Applying norm at every step introduces ~5e-4 of FP error per accumulation,
        // which, once it crosses an i8 threshold, is amplified to ~1.5e-2 at the L2 output.
        var acc: f32     = 0.0;

        // CfC part: channels 0..96 (w_cfc holds filtered_retina).
        // ANTI-FMA: store the product in a var before adding, which prevents fusion into
        // an FMA (a GPU may fuse a*b+c into an FMA and get a different result from the CPU,
        // which does it separately: product = a*b; acc += product. Forcing the same order
        // guarantees bit-for-bit agreement.)
        if l1_active {
            for (var i: u32 = 0u; i < L1_OUT; i = i + 1u) {
                if life_edge_enabled(being, 1u, j, i) {
                    let prod: f32 = buf_l1w[l1_base + i] * f32(w_cfc[i]);
                    acc = acc + prod;
                }
            }
        }
        // KAN-only part: channels 96..WAVE (they went through the KAN but not the CfC).
        if l1_active {
            for (var i: u32 = L1_OUT; i < WAVE; i = i + 1u) {
                if life_edge_enabled(being, 1u, j, i) {
                    let prod: f32 = buf_l1w[l1_base + i] * f32(w_kan[i]);
                    acc = acc + prod;
                }
            }
        }

        // Normalise by 127 exactly once — identical to the CPU (layers.rs:49).
        w_l1r[j] = select(acc * (1.0 / 127.0), 0.0, !l1_active);
        if observes(being) {
            buf_observation[OBS_L1_MATMUL + j] = bitcast<u32>(w_l1r[j]);
        }

        // DEBUG (being 0 only): exports l1_raw plus the CfC/KAN inputs for diagnostics.
        // buf_debug layout: [l1_raw[0..96], cfc_in[0..96], kan_in[96..223]]
        if being == 0u {
            // l1_raw[j]
            buf_debug[j] = w_l1r[j];
            // CfC inputs (channels 0..96)
            if j < L1_OUT {
                buf_debug[96u + j] = f32(w_cfc[j]);
            }
            // KAN inputs (channels 96 onwards). The `j < 127u` bound is a leftover from
            // the 223-channel wave; only 96 threads reach this line, so with WAVE=192 it
            // writes exactly the 96 KAN-only channels and never binds.
            if j < 127u {
                buf_debug[192u + j] = f32(w_kan[j + 96u]);
            }
        }
    }

    workgroupBarrier();

    // Quantisation l1_raw → l1_act (engine.rs:461-463), extended with P1.7:
    //   l1_act[i] = clamp(ACTIVATION(MEMORY(PERIOD/THRESHOLD(l1_raw[i]))) × 127, -127, 127)
    //
    // The four steps (mechanism 2 → 1 → 3) live in the SAME `if tid < L1_OUT` block that
    // already existed (they open no new barrier): each thread only reads and writes its
    // OWN index in `w_l1r`/`buf_neuron_mem` — no cross-neuron dependency — so they fit
    // inside the quantisation block with no risk of a race. `w_l1r[tid]` is OVERWRITTEN
    // with the post-mechanism value: the L1 bypass (STAGE 2, threads 48..95, further
    // below) reads the SAME array and therefore inherits the effect automatically — which
    // is correct, because memory, period and threshold are properties of the NEURON, not
    // of "where the signal goes next".
    if tid < L1_OUT {
        let j: u32 = tid;
        let meta_base: u32 = being * NEURON_META_PER_BEING;
        let mem_base:  u32 = being * NEURON_MEM_PER_BEING;
        let off = params.mech_off_mask;

        // ── Mechanism 2 (period) — decides whether THIS tick recomputes the memory ──
        // HONESTY (project CLAUDE.md / the task): on the GPU all 96 threads run anyway on
        // every dispatch — a period > 1 does NOT save GPU time, it only decides WHETHER
        // the memory is updated with this tick's x[n] or keeps the previous value. The
        // gain is in REPRESENTATION (time scales coexisting — the owner's original
        // request), not in speed.
        let period_raw: f32 = buf_neuron_meta[meta_base + NEURON_META_L1_PERIOD_OFF + j];
        let period_off: bool = (off & 2u) != 0u;
        // The +0.5 before the u32() truncation rounds instead of truncating downwards (a
        // period stored as 2.999999 through floating-point error would otherwise become 2).
        // max(...,1u): a period of 0 (an uninitialised buffer, e.g. `forward()` /
        // `gpu_validate`, which never upload this buffer) falls back to "recompute every
        // tick", matching today's behaviour — never "never recompute".
        let period: u32 = select(max(u32(period_raw + 0.5), 1u), 1u, period_off);
        // counter_new was already written into buf_cfc by tid==0 in STAGE 0B, before the
        // first workgroupBarrier() — so it is visible here to all 96 threads.
        let tick_counter: u32 = bitcast<u32>(buf_cfc[being * 97u + 96u]);
        let fires: bool = period <= 1u || (tick_counter % period) == 0u;

        // ── Mechanism 1 (memory) — an EMA with amplitude gain ───────────────────
        // The full derivation is in `neuron_mem_gain`, above the kernel.
        let alpha_raw: f32 = buf_neuron_meta[meta_base + j];
        let mem_off: bool = (off & 1u) != 0u;
        let alpha: f32 = select(alpha_raw, 0.0, mem_off);
        var h: f32 = buf_neuron_mem[mem_base + j];
        if fires {
            // x[n] = this tick's l1_raw (the STAGE 1 matmul, still untouched here).
            h = h * alpha + w_l1r[j] * (1.0 - alpha);
            buf_neuron_mem[mem_base + j] = h;
        }
        // Outside the neuron's own tick (!fires), `h` is already the value stored the last
        // time it fired — "keep the previous value", without touching the buffer.
        var v_eff: f32 = h * neuron_mem_gain(alpha);

        // A capacity-masked neuron does not retain a stale contribution while it is
        // dormant. It can re-enter cleanly as age/energy capacity grows.
        if !life_neuron_active(being, 1u, j) {
            v_eff = 0.0;
            buf_neuron_mem[mem_base + j] = 0.0;
        }

        // ── Mechanism 2 (threshold) — below it, the contribution is zero ────────
        let thr_raw: f32 = buf_neuron_meta[meta_base + NEURON_META_L1_THRESHOLD_OFF + j];
        let threshold_off: bool = (off & 4u) != 0u;
        let threshold: f32 = select(thr_raw, 0.0, threshold_off);
        if abs(v_eff) < threshold { v_eff = 0.0; }

        // The bypass (STAGE 2, tid 48..95) and the quantisation below read the final value.
        w_l1r[j] = v_eff;
        if observes(being) {
            buf_observation[OBS_L1_EFFECTIVE + j] = bitcast<u32>(v_eff);
        }

        // ── Mechanism 3 (heterogeneous activation) ──────────────────────────────
        let act_raw: f32 = buf_neuron_meta[meta_base + NEURON_META_L1_ACTIVATION_OFF + j];
        let activation_off: bool = (off & 8u) != 0u;
        let act_id: u32 = select(u32(act_raw + 0.5), 0u, activation_off);
        let activated: f32 = apply_activation(act_id, v_eff);
        // M4: the multiplicative gate of this neuron's REGION (block = tid/24).
        // 1.0 is neutral (flag off). It gates the quantised L2 input. The historical
        // bypass below reads `w_l1r`, so it is intentionally described separately.
        let gate: f32 = buf_gates[being * 4u + tid / 24u];
        let l1_emission = i32(clamp(activated * gate * 127.0, -127.0, 127.0));
        w_l1a[tid] = select(l1_emission, 0, output_is_ablated(being, ABLATION_L1_WORD, tid));
        if observes(being) {
            buf_observation[OBS_L1_TO_L2 + tid] = bitcast<u32>(w_l1a[tid]);
        }
    }

    workgroupBarrier();

    // ─────────────────────────────────────────────────────────────────────────
    // STAGE 2: FloatLinear L2 — 96→48 (float_layers.rs:23-34)
    //          + the L1 bypass into final_intent[48..96]
    //
    // ENGINE (engine.rs:465-472):
    //   final_intent[0..48]  = tanh(l2_raw × out_gain)         — through L2
    //   final_intent[48..96] = tanh(l1_raw[48..96] × out_gain) — bypassing L2
    //   (`out_gain` used to be the hardcoded constant `2.0`; see `params.out_gain` — P1.5)
    //
    // Why a bypass? L2 has 48 outputs but final_intent has 96 slots. Slots 48..95 are
    // filled directly with l1_raw (pre-L2).
    //
    // Threads 0..47:  compute L2 and write final_intent[0..47]
    // Threads 48..95: write the bypass into final_intent[48..95]
    // Threads 96..127: idle
    //
    // buf_l2w layout: being*(L2_OUT*L1_OUT) + j*L1_OUT + i
    // ─────────────────────────────────────────────────────────────────────────

    let intent_base: u32 = being * 96u;

    // ── Sub-stage 2a: L2 matmul, storing the PRE-gain l2_raw in LDS ────────────
    //
    // Separated from the rest (fatigue + tanh) because the std-dev normalisation (item 3
    // of P1.5, `params.norm_l2_by_std`) needs all 48 l2_raw values of the SAME being ready
    // at once — each thread only computes ITS neuron j, but the standard deviation is over
    // all 48. `w_l2r` holds the same quantity `float_layers.rs::forward` returns on the
    // CPU (acc/127, with NO gain at all).
    if tid < L2_OUT {
        let j: u32 = tid;
        let l2_base: u32 = being * (L2_OUT * L1_OUT) + j * L1_OUT;
        let l2_active = life_neuron_active(being, 2u, j);
        // FIX: accumulate without a per-step norm — apply 1/127 once at the end
        // (float_layers.rs:29-34: acc += w[i] * input[i]; output = acc * norm_factor)
        var acc: f32     = 0.0;

        // ANTI-FMA: same pattern as L1 — store the product in a var before adding.
        if l2_active {
            for (var i: u32 = 0u; i < L1_OUT; i = i + 1u) {
                if life_edge_enabled(being, 2u, j, i) {
                    let prod: f32 = buf_l2w[l2_base + i] * f32(w_l1a[i]);
                    acc = acc + prod;
                }
            }
        }

        let raw_l2: f32 = acc * (1.0 / 127.0);
        if observes(being) {
            buf_observation[OBS_L2_MATMUL + j] = bitcast<u32>(raw_l2);
        }

        // ── P1.7, mechanism 1 (memory) for L2 — with no period and no threshold: the task
        // asks for period/threshold on L1 only ("real asynchrony" targets the SENSORY
        // input, where different time scales make physical sense; L2 already reads all of
        // L1 every tick). L2 always "fires" — `fires` is implicit here. Same gain
        // derivation as `neuron_mem_gain` above the kernel.
        let l2_meta_base: u32 = being * NEURON_META_PER_BEING + NEURON_META_L2_ALPHA_OFF;
        let l2_mem_base:  u32 = being * NEURON_MEM_PER_BEING + L1_OUT;  // slots [96..144) = L2
        let alpha_l2_raw: f32 = buf_neuron_meta[l2_meta_base + j];
        let mem_off_l2: bool = (params.mech_off_mask & 1u) != 0u;
        let alpha_l2: f32 = select(alpha_l2_raw, 0.0, mem_off_l2);
        var h_l2: f32 = buf_neuron_mem[l2_mem_base + j];
        h_l2 = h_l2 * alpha_l2 + raw_l2 * (1.0 - alpha_l2);
        buf_neuron_mem[l2_mem_base + j] = h_l2;

        w_l2r[j] = select(h_l2 * neuron_mem_gain(alpha_l2), 0.0, !l2_active);
        if !l2_active {
            buf_neuron_mem[l2_mem_base + j] = 0.0;
        }
        if observes(being) {
            buf_observation[OBS_L2_EFFECTIVE + j] = bitcast<u32>(w_l2r[j]);
        }
    }

    // A barrier under UNIFORM CONTROL — ALL 128 threads reach it, including those that did
    // not write w_l2r (tid >= 48), because WGSL requires workgroupBarrier() not to sit
    // inside an `if` that only some threads execute (otherwise the threads that skip the
    // `if` never reach the barrier and the driver hangs or diverges — the barrier MUST be
    // in non-divergent code).
    workgroupBarrier();

    // ── Sub-stage 2b: population standard deviation of this being's 48 l2_raw ──
    //
    // This only comes into play when `params.norm_l2_by_std == 1`; with the flag OFF (the
    // default) `l2_std` stays at 1.0 and the division below is an exact no-op — it does
    // not change a single bit of the historical result. Every thread recomputes the same
    // value (redundant, but 48 additions are negligible next to the 96×48 matmul above) —
    // cheaper than another workgroup array plus another barrier just to broadcast a scalar.
    //
    // POPULATION standard deviation (divisor N, not N-1): the purpose here is to normalise
    // amplitude, not to infer a sample's variance — the same spirit as layer norm. It does
    // NOT subtract the mean: the hypothesis under test is that the SATURATION comes from
    // the SCALE of l2_raw growing with mutation across generations
    // (docs/PLANO_EVOLUCAO.md P1.5), not from a directional bias — subtracting the mean
    // would also remove any genuine directional signal (e.g. "it turns right more than
    // left on this track"), which is not what we want to attack here.
    var l2_std: f32 = 1.0;
    if params.norm_l2_by_std == 1u {
        var sum: f32 = 0.0;
        var sum_sq: f32 = 0.0;
        for (var k: u32 = 0u; k < L2_OUT; k = k + 1u) {
            let v: f32 = w_l2r[k];
            sum = sum + v;
            sum_sq = sum_sq + v * v;
        }
        let mean: f32 = sum / f32(L2_OUT);
        // max(0.0, …): a variance is never negative by definition, but computing it as a
        // sum of squares (E[x²] - E[x]²) can produce a negative epsilon through
        // floating-point cancellation when the 48 values are nearly identical.
        let variance: f32 = max(0.0, sum_sq / f32(L2_OUT) - mean * mean);
        // NORM_EPS avoids an explosion at the start of training, when the 48 l2_raw are
        // still almost all ~0 (freshly drawn weights, σ=0.30 diluted across 96 inputs) —
        // dividing by a near-zero deviation would amplify numerical NOISE, not signal.
        // 1e-3 sits well below any genuine deviation observed (a typical saturated l2_raw
        // is ~3.8, see the OUT_BETA comment above).
        const NORM_EPS: f32 = 1e-3;
        l2_std = max(sqrt(variance), NORM_EPS);
    }
    if tid == 0u && observes(being) {
        buf_observation[OBS_L2_STD] = bitcast<u32>(l2_std);
    }

    if tid < L2_OUT {
        // Thread tid ∈ [0,47]: L2 neuron j
        let j: u32 = tid;

        // Normalise by the SCALE (with the flag off, l2_std=1.0 and the division is a
        // no-op), then apply the output gain — the same order of magnitude that
        // `acc * (2.0/127.0)` had before, when out_gain=2.0 and normalisation is OFF:
        // w_l2r[j] * out_gain == acc*(1/127)*2.0 == acc*(2.0/127.0).
        let l2_scaled: f32 = w_l2r[j] / l2_std;
        let l2_raw: f32 = l2_scaled * params.out_gain;

        // Output fatigue: reads slot [96+j] of buf_fatigue (the second half of the widened
        // buffer). It tracks |intent| for channel j and subtracts params.out_beta *
        // out_fat from raw before the tanh. This breaks the output fixed point without
        // affecting channels that vary normally.
        let out_fat_base: u32 = being * L1_OUT * 2u + L1_OUT;  // slots [96..192] per being
        var out_fat: f32 = buf_fatigue[out_fat_base + j];
        out_fat = out_fat + OUT_ALPHA * (abs(l2_raw) - out_fat) * params.dt;
        buf_fatigue[out_fat_base + j] = out_fat;

        // Output fatigue as a SOFT THRESHOLD on the MAGNITUDE:
        //   intent = tanh(sign(raw) * max(0, |raw| - params.out_beta * out_fat))
        // `out_fat` tracks |raw|, so the earlier form (`raw - OUT_BETA*out_fat`) was
        // ASYMMETRIC: with a sustained negative raw it subtracted a positive term from an
        // already-negative value and pushed it into saturation (-0.999), while with a
        // positive raw it attenuated to 0.15·raw. The fatigue biased the whole population
        // downwards. It now shrinks the magnitude and preserves the sign, on both sides.
        let adapted: f32 = sign(l2_raw) * max(0.0, abs(l2_raw) - params.out_beta * out_fat);
        if observes(being) {
            buf_observation[OBS_L2_ADAPTED + j] = bitcast<u32>(adapted);
        }

        // P1.7, mechanism 3 — L2's heterogeneous activation (the FINAL non-linearity of
        // L2, previously hardcoded as tanh). The L1 bypass (below) keeps a fixed tanh — it
        // is MOTOR OUTPUT squashing, not "the L1 neuron's activation" (which was already
        // decided at the quantisation, STAGE 1).
        let l2_act_base: u32 = being * NEURON_META_PER_BEING + NEURON_META_L2_ACTIVATION_OFF;
        let act_raw_l2: f32 = buf_neuron_meta[l2_act_base + j];
        let activation_off_l2: bool = (params.mech_off_mask & 8u) != 0u;
        let act_id_l2: u32 = select(u32(act_raw_l2 + 0.5), 0u, activation_off_l2);
        // clamp: the comment on apply_activation promised "it saturates in the caller's
        // clamp", but no caller clamped — the linear activation (id=2) leaked a RAW value
        // (measured in the lab: slots 16/31/45 of the champion reaching up to 3.19) and
        // inflated every metric that assumes intent ∈ [-1,1] (saturation, dims_sat,
        // intent_mag). The motor channels (0..3) were never affected: kan_eval/context
        // clamp them.
        let l2_emission = clamp(apply_activation(act_id_l2, adapted), -1.0, 1.0);
        buf_intent[intent_base + j] = select(
            l2_emission,
            0.0,
            output_is_ablated(being, ABLATION_L2_WORD, j),
        );
    }

    // Threads 48..95: the L1 bypass → final_intent[48..95]
    // (engine.rs:469-471): final_intent[i] = tanh(l1_raw[i] × out_gain) for i ∈ [48,95].
    // The std-dev normalisation (params.norm_l2_by_std) does NOT apply here — the name is
    // "norm_l2_by_std" because it measures and corrects the scale of l2_raw specifically;
    // the bypass does not go through the L2 matmul, so there is no "l2_raw" to normalise.
    if tid >= L2_OUT && tid < L1_OUT {
        // The same output-fatigue mechanism, applied to the bypass.
        let raw_bp: f32 = w_l1r[tid] * params.out_gain;
        let out_fat_base_bp: u32 = being * L1_OUT * 2u + L1_OUT;
        var out_fat_bp: f32 = buf_fatigue[out_fat_base_bp + tid];
        out_fat_bp = out_fat_bp + OUT_ALPHA * (abs(raw_bp) - out_fat_bp) * params.dt;
        buf_fatigue[out_fat_base_bp + tid] = out_fat_bp;

        let adapted_bp: f32 = sign(raw_bp) * max(0.0, abs(raw_bp) - params.out_beta * out_fat_bp);
        if observes(being) {
            buf_observation[OBS_BYPASS_ADAPTED + tid - L2_OUT] = bitcast<u32>(adapted_bp);
        }
        let bypass_emission = tanh(adapted_bp);
        buf_intent[intent_base + tid] = select(
            bypass_emission,
            0.0,
            output_is_ablated(being, ABLATION_L1_WORD, tid),
        );
    }

    // ── OPTIONAL LIFETIME PLASTICITY ────────────────────────────────────────
    // Eligibility is causal: it uses the latent and motor from the previous
    // transition, then the current feedback modulates the bounded P update.  The
    // update uses elapsed world time, including ticks skipped by brain_skip.
    if (params.lifetime_flags & LIFE_PLASTICITY) != 0u {
        let life = life_base(being);
        let elapsed = life_step_elapsed(being);
        if tid < L2_OUT {
            let j = tid;
            let previous_latent = buf_lifetime[life + LIFE_PREV_LATENT + j];
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                let e_index = life + LIFE_E + k * L2_OUT + j;
                let decay = exp(-clamp(life_gene(being, 1u), 0.0, 16.0) * elapsed);
                let old_e = buf_lifetime[e_index];
                let old_motor = buf_lifetime[life + LIFE_PREV_MOTOR + k];
                let outer = previous_latent * old_motor;
                buf_lifetime[e_index] = old_e * decay + outer * (1.0 - decay);
            }
        }
        storageBarrier();
        workgroupBarrier();

        if tid < 3u {
            let k = tid;
            let energy = clamp(life_feedback(being, 1u), 0.0, 1.0);
            let alive = clamp(life_feedback(being, 2u), 0.0, 1.0);
            let energy_exponent = clamp(life_gene(being, 15u), 0.25, 4.0);
            let energy_term = pow(energy, energy_exponent);
            let modulation = life_feedback(being, 0u) * clamp(life_gene(being, 12u), 0.0, 4.0)
                * alive * energy_term;
            let learning = clamp(life_gene(being, 0u), 0.0, 1.0) * elapsed * modulation;
            let p_clamp = clamp(life_gene(being, 2u), 0.001, 1.0);
            let gain = clamp(life_gene(being, 3u), 0.0, 1.0);
            var residual = 0.0;
            for (var j: u32 = 0u; j < L2_OUT; j = j + 1u) {
                let p_index = life + LIFE_P + k * L2_OUT + j;
                let e_index = life + LIFE_E + k * L2_OUT + j;
                let retention = exp(-clamp(life_gene(being, 13u), 0.0, 4.0) * elapsed);
                let updated = clamp(buf_lifetime[p_index] * retention + learning * buf_lifetime[e_index], -p_clamp, p_clamp);
                buf_lifetime[p_index] = updated;
                residual = residual + updated * clamp(w_l2r[j], -1.0, 1.0);
            }
            residual = clamp(residual / f32(L2_OUT), -p_clamp, p_clamp) * gain;
            var motor = clamp(buf_intent[intent_base + k] + residual, -1.0, 1.0);
            // Ablation is the final authority, including over a plastic residual.
            if output_is_ablated(being, ABLATION_L2_WORD, k) {
                motor = 0.0;
            }
            buf_intent[intent_base + k] = motor;
            buf_lifetime[life + LIFE_PREV_MOTOR + k] = motor;
            buf_lifetime[life + LIFE_RUNTIME + 9u + k] = residual;
        }
        if tid < L2_OUT {
            buf_lifetime[life + LIFE_PREV_LATENT + tid] = clamp(w_l2r[tid], -1.0, 1.0);
        }
        workgroupBarrier();
    }

    // Note: no barrier is needed at the end — every thread writes to a disjoint position
    // in buf_intent and nothing reads it back inside this dispatch.
}
