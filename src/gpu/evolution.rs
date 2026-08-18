//! Genome selection and mutation between generations — runs on the CPU.
//!
//! The COST is in the ticks (GPU). Evolution happens between generations,
//! once per N ticks. Therefore it can run on the CPU with no significant penalty.
//!
//! # Genome
//!
//! A `Genome` carries the L1 (BitLinear f32) and L2 (FloatLinear f32) weights of the
//! `SovereignEngine`, plus the output KAN knots. They are the same fields as the
//! main engine's `EliteDna`, but without the serialization metadata.
//!
//! # Evolutionary algorithm
//!
//! - **Selection**: tournament (k=3) — selects the best among 3 random candidates.
//!   Less prone to genetic drift than proportional selection.
//! - **Gaussian mutation**: N(0, σ) noise per weight, with σ=0.05 by default.
//!   Each weight has probability `p_mutate` of being perturbed.
//! - **Elitism**: the best `n_elite` genomes pass through without mutation.
//!   Preserves progress; σ covers exploration.
//!
//! # Random initialization
//!
//! `Genome::random()` uses an LCG with a seed derived from the being's index. Magnitude σ=0.3
//! for L1/L2. Zeroed weights (BitLinear::new()) → null network → no movement;
//! σ=0.3 puts weights at the scale motor.rs uses (~0.1–0.5 after training).
//!
//! # Spectral initialization of L1 (P0.5)
//!
//! The i.i.d. N(0, σ=0.30) draw above, measured with `analysis::activation` in 4
//! independent runs, produces an L1 spectral gain (largest singular value of the
//! 192→96 feedforward matrix, the same quantity that `AlarmKind::SpectralRadiusHigh`
//! audits) of ~7.0 — well above the `|λ_max| <= ~1` that echo-state theory asks
//! for so the sensory signal is not amplified before the first mutation. After
//! drawing `l1`, `Genome::random` measures this gain with
//! `math::linalg::largest_singular_value` and rescales ALL of L1's weights by a
//! single scalar factor (`thresholds::SPECTRAL_INIT_TARGET / measured`) — a scalar
//! preserves the relative direction of each weight, only adjusting magnitude, and since the measure is
//! LINEAR in the matrix (`largest_singular_value(c·W) = c · largest_singular_value(W)`,
//! see test `largest_singular_value_escala_linear_com_a_matriz` in
//! `math/linalg.rs`), the factor hits the target exactly, not approximately.
//!
//! Controlled by [`SPECTRAL_INIT_ON`] (default: on) — the A/B control arm is
//! `--sem-init-espectral` in `gpu_evolve`. It only touches `l1`: L2 does not enter the alarm
//! (`collect_activation_stats("L2", ..., w_rec: None, ...)` in `gpu_evolve/main.rs`),
//! so there is no "what the alarm measures" for L2 to target. `load_genome` (genome coming from
//! disk) NEVER goes through `Genome::random` — it builds the `Genome` directly via struct
//! literal — so a champion saved before this change loads with the spectral gain
//! it always had, untouched.

use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
use crate::engine::SovereignEngine;
use crate::kan::KAN_OUT_KNOTS;
use crate::math::{linalg, thresholds};
use std::sync::atomic::{AtomicBool, Ordering};

/// Turns L1 spectral initialization (P0.5) on/off in `Genome::random`.
///
/// Default `true` (ON) — being born with spectral gain ≈ `thresholds::SPECTRAL_INIT_TARGET`
/// is the recommended behavior, not a side experiment. `gpu_evolve
/// --sem-init-espectral` turns it off (`store(false, ...)`) to run the A/B control
/// arm: a population born with the raw i.i.d. σ=0.30, no correction — the same brain
/// from before hypothesis P0.5.
///
/// It is a global `AtomicBool` (not a parameter of `Genome::random`) for the same reason as
/// `world::NO_HOLES`/`world::EASY_TRACK`: `Genome::random(seed)` is called in several
/// dozen places (initial population, checkpoint resume, tests) and changing the
/// signature would force touching all of them just to propagate a CLI flag that is read
/// once at process start.
pub static SPECTRAL_INIT_ON: AtomicBool = AtomicBool::new(true);

/// Scalar factor that takes a `medido` (measured) gain to `alvo` (target) (== `alvo / medido`).
///
/// Pure function, no global state — extracted from `Genome::random` so it can be
/// tested without depending on [`SPECTRAL_INIT_ON`] (which is global and mutable; a test that
/// toggled it would risk interfering with any other test calling
/// `Genome::random` in parallel, see R8 of the project's CLAUDE.md — "canary: identical
/// numbers where they should diverge").
///
/// `medido <= 1e-6` (entire L1 ~0, never observed with σ=0.30 and 18.432 weights) returns
/// `1.0` — nothing to scale, and `0.0 × 1.0 = 0.0` stays null, not a division by
/// near-zero blowing up the weight.
fn spectral_rescale_factor(medido: f32, alvo: f32) -> f32 {
    if medido > 1e-6 { alvo / medido } else { 1.0 }
}

// ── Derived dimensions ────────────────────────────────────────────────────────
pub const N_L1_WEIGHTS: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 192 × 96 = 18.432
pub const N_L2_WEIGHTS: usize = L1_NEURONS * L2_NEURONS; // 96  × 48 =  4.608
pub const N_KAN_KNOTS: usize = 3 * KAN_OUT_KNOTS; // 3   × 12 =     36

/// Complete genome of a being: L1 + L2 + output KAN.
#[derive(Clone)]
pub struct Genome {
    /// L1 weights (instinct_layer): SENSORY_WAVE_SIZE × L1_NEURONS f32.
    /// Typical post-training magnitude: ~0.1–0.5. Random init with σ=0.3.
    pub l1: Vec<f32>,
    /// L2 weights (cortex_l2): L1_NEURONS × L2_NEURONS f32.
    pub l2: Vec<f32>,
    /// Output KAN knots: 3 outputs × KAN_OUT_KNOTS f32.
    /// Init with KanOutputHead::new() (tanh ramp — equivalent to a smooth identity).
    pub kan_knots: [[f32; KAN_OUT_KNOTS]; 3],
    /// SLOW brain weights in `SlowBrain::all_weights()` order:
    /// dense1.weights(1024) | dense1.bias(16) | dense2.weights(48) | dense2.bias(3).
    ///
    /// Without this, the slow brain is not an organ, it is a constant: `SlowBrain::new()` uses
    /// FIXED seeds, so all beings would be born with the same slow weights and
    /// modulation would be identical for the entire population — evolution would have no way
    /// to tune it nor to get rid of it. Heritable and mutable, it can be selected
    /// or discarded by the process itself.
    pub slow: Vec<f32>,
    /// τ bank: CfC decay rate per channel (s⁻¹), `N_CFC_CHANNELS` values.
    ///
    /// Born uniform at `DECAY_DEFAULT` (τ = 2.0 s, the previous single value) and only becomes
    /// heterogeneous via `randomize_tau`. This way the control condition is literally the
    /// historical brain running through the new code — same shader, same formula, only the
    /// vector differs.
    pub tau: Vec<f32>,
    /// Bank of heritable sensors (SENSORS PHASE, step 1) — `N_SENSOR_GENES` values
    /// in `sensor_idx` order: FOV of the wall/curvature sweep + wall range +
    /// lateral range + 3 curvature lookaheads. All live in `perception.wgsl`
    /// (spatial pipeline) — none changes the SIZE of a GPU buffer, only the VALUE that was
    /// already read from a compile-time constant.
    ///
    /// Born at `SENSOR_DEFAULTS` (today's hardcodes, byte-identical to the shader's
    /// literals) and only becomes heterogeneous when `SENSORS_EVOLVE_ON` is on — same
    /// precedent as the τ bank, which is the mold for this change (see the project's CLAUDE.md,
    /// Leonardo's request: "copy its pattern"): the experiment's control arm is
    /// literally today's eyes, not a third behavior.
    pub sensors: Vec<f32>,

    // ── P1.7 — asynchronous brain (docs/PLANO_EVOLUCAO.md §P1.7) ────────────────
    //
    // Leonardo's critique that motivated the three fields below: *"what I am asking
    // for is an ASYNCHRONOUS brain; it will work in different dimensions, neurons and parts
    // depending on where it is on the track; if you always stay in the same tanh you are just
    // burning time."* Measured (not assumed): 20 uses of tanh in `brain.wgsl` and
    // no other activation; the 96 L1 neurons and 48 L2 neurons fire on EVERY
    // tick. The CfC τ bank (`tau`, above) is already heritable/mutable and is the
    // PRECEDENT copied here — when it became heterogeneous, it broke the Universe's
    // 680 m wall.
    //
    // All six are born NEUTRAL (identical to today's brain) in
    // `Genome::random`/`from_engine` — only MUTATION in `evolve()` moves them away from
    // neutral, and only when the `NEURON_*_MUTATE_ON` atomics (default `true`, below)
    // allow it. `gpu_evolve --sem-X` turns off the corresponding atomic (mutation
    // freezes at neutral for the rest of the run) AND forces `BrainGpu::*_off = true`
    // (the shader ignores whatever the genome carries, even if inherited from a
    // `--retomar` predating the flag) — the strongest possible control arm.
    /// Mechanism 1 (per-neuron memory): alpha of the EMA filter
    /// `h[n] = h[n-1]·alfa + x[n]·(1-alfa)`, one per neuron of L1(96)+L2(48).
    /// Born 0.0 (no memory = identical to today). Clamp in `evolve()`: [0, 0.95] —
    /// alfa≥1 diverges (stability condition of the difference equation, the filter's
    /// pole has to stay inside the unit circle). Layout: `[0..96)`=L1,
    /// `[96..144)`=L2 — same order as `BrainGpu::NEURON_MEM_PER_SER` on the host.
    pub neuron_alpha: Vec<f32>,
    /// Mechanism 2 (period): every how many ticks the L1 neuron recomputes its
    /// memory (outside its tick, it keeps the previous value — reuses mechanism 1's
    /// buffer). Born 1.0 (recomputes every tick = identical to today). Integer
    /// stored as f32 (same convention as every genome field — disk is
    /// positional, f32 only, see `genome_io.rs`). L1 only — only there does temporal
    /// asynchrony make physical sense (sensory input, see the mechanism doc in
    /// `brain.wgsl`); L2 already reads all of L1 every tick.
    pub neuron_period_l1: Vec<f32>,
    /// Mechanism 2 (threshold): below it, the L1 neuron contributes zero. Born
    /// 0.0 (always contributes = identical to today, since the post-activation
    /// magnitude is never negative). L1 only, same reason as the period.
    pub neuron_threshold_l1: Vec<f32>,
    /// Mechanism 3 (heterogeneous activation): activation id per neuron of
    /// L1(96)+L2(48). 0=tanh 1=ReLU 2=linear 3=leaky-ReLU(0.01) — see
    /// `math::activations` for the formulas and `brain.wgsl::apply_activation` for
    /// the GPU mirror. Born 0.0 (tanh = identical to today). Layout: `[0..96)`=L1,
    /// `[96..144)`=L2.
    pub neuron_activation: Vec<f32>,
    /// M4 (REGIOES_CEREBRAIS.md): conductor weights — 4 region gates × (3 outputs
    /// of the slow brain + bias) = 16 f32. gate_b = 1 + 0.5·tanh(w·slow_out + b).
    /// Heritable and mutated like the slow brain. OPTIONAL block on disk (genome_io).
    pub gates_w: Vec<f32>,
}

/// L1 and L2 neurons heritable by the P1.7 mechanisms — same numbers as
/// `config::iara::brain::{L1_NEURONS, L2_NEURONS}`, re-exported here so
/// `genome_io.rs` can size the disk blocks without duplicating the import.
pub const N_L1_NEURONS: usize = L1_NEURONS;
pub const N_L2_NEURONS: usize = L2_NEURONS;

/// P1.7 — control arm of mechanism 1's MUTATION (memory). Default `true`:
/// `evolve()` mutates `neuron_alpha` like any other heritable field (same
/// precedent as the τ bank, mutated unconditionally). `gpu_evolve
/// --sem-memoria-neuronal` turns it off — the field stays FROZEN at 0.0 (the birth
/// value) for the rest of the run, and `BrainGpu::mem_off` is turned on along with it (the shader
/// ignores the field even if an earlier `--retomar` had already mutated something).
pub static NEURON_MEM_MUTATE_ON: AtomicBool = AtomicBool::new(true);
/// P1.7 — same for mechanism 2, period sub-part (`--sem-periodo-neuronal`).
pub static NEURON_PERIOD_MUTATE_ON: AtomicBool = AtomicBool::new(true);
/// P1.7 — same for mechanism 2, threshold sub-part (`--sem-limiar-neuronal`).
pub static NEURON_THRESHOLD_MUTATE_ON: AtomicBool = AtomicBool::new(true);
/// P1.7 — same for mechanism 3, heterogeneous activation (`--sem-ativacao-heterogenea`).
pub static NEURON_ACTIVATION_MUTATE_ON: AtomicBool = AtomicBool::new(true);

/// Number of heritable sensory parameters (SENSORS PHASE, step 1).
///
/// Covers only the CHEAP sensors: changing the VALUE does not change the size of any GPU buffer,
/// only the formula that already exists in `perception.wgsl` (spatial pipeline — `buf_sensors`,
/// binding 7 of `layout_spatial`). Fixed order, used in `Genome::random`,
/// `pack_sensors`, `save_genome`/`load_genome` and in the shader (`perception.wgsl`, the same
/// `N_SENSOR_GENES` constant on the GPU side) — both sides have to agree on the order:
/// there is no field name on disk nor in the buffer, only position (same rule as the whole
/// genome, see `genome_io.rs`).
///
/// OUT OF SCOPE in this phase (exploration map that motivated this change):
/// - **Retina resolution / neuron count**: they change the SIZE of a GPU buffer (the
///   genome itself, in the case of L1/L2) — that is not a "sensor", it is the SHAPE of the genome. The
///   `.bin` format is positional; changing this invalidates every saved `.bin`.
/// - **Retina range** (`visao::Params`, `ALCANCE_PADRAO`): the mechanism already exists
///   (uniform + its own tests) but `gpu_evolve` NEVER calls `visao::definir()` — only the
///   Bevy/UI binary (`main.rs`) calls it. Inheriting this now would create a DEAD gene in
///   the training binary (gap 10 of CLAUDE.md: "state computed and never read"). Connecting
///   `gpu_evolve` to `visao::Params` is a prerequisite of a future step, not this one.
/// - **Whisker range** (`WHISKER_RANGE`): duplicated in `motor.rs` (CPU/Bevy, outside
///   `gpu_evolve`'s path) and `perception.wgsl` (GPU) with no shared constant.
///   Inheriting only the GPU side would make the training binary diverge from the visualizer
///   with no warning at all — deferred until a single source exists between the two paths.
/// - **Curvature 200/260/275/350 m** (`perception_temporal.wgsl`, ch101/111-113, and
///   `motor.rs::curv_at(260.0)` on the CPU path): they live in a DIFFERENT PIPELINE
///   (`layout_temporal`, not `layout_spatial`) — they enter as the NEXT sensory gene,
///   same recipe, new buffer in that layout. Not included here to keep the
///   first cut small enough to actually review.
pub const N_SENSOR_GENES: usize = 6;

/// Sensory gene indices — same order in `SENSOR_DEFAULTS`, `SENSOR_RANGES`,
/// `perception.wgsl` (`N_SENSOR_GENES`/`sensors[being * N_SENSOR_GENES + i]`) and in the
/// `.bin` file on disk.
pub mod sensor_idx {
    pub const FOV_HALF_RAD: usize = 0;
    pub const WALL_RANGE_M: usize = 1;
    pub const LAT_RANGE_M: usize = 2;
    pub const CURV_NEAR_M: usize = 3;
    pub const CURV_MID_M: usize = 4;
    pub const CURV_FAR_M: usize = 5;
}

/// Birth value of each sensory gene — TODAY's hardcodes, byte-identical to the
/// literals in `perception.wgsl` (old `FOV_HALF_RAD`, `WALL_MAX`, `LAT_SAMPLES[9]` and
/// the 3 `dists` of BLOCK K). Used when `SENSORS_EVOLVE_ON` is off (default) and
/// as `load_genome`'s fallback for a `.bin` written before this field existed — in both
/// cases the goal is to reproduce today's behavior, not invent one.
pub const SENSOR_DEFAULTS: [f32; N_SENSOR_GENES] = [
    0.872664625997, // fov_half_rad: 50° — literal identical to the old perception.wgsl
    60.0,           // wall_range_m: today's WALL_MAX
    40.0,           // lat_range_m: today's LAT_SAMPLES[9] (the maximum sample range)
    40.0,           // curv_near_m: BLOCK K dists[0]
    70.0,           // curv_mid_m:  BLOCK K dists[1]
    120.0,          // curv_far_m:  BLOCK K dists[2]
];

/// PHYSICALLY valid ranges of each gene — used in the initial draw (when evolution
/// is on, `random_sensors_within_ranges`) and as a CLAMP after every
/// mutation (`mutate_sensors`), so an impossible sensor is never produced.
///
/// # Where each range comes from
///
/// - **fov_half_rad** (½-angle 30°..85°, total aperture 60°..170°): the example
///   range requested by the task. Floor of 60° aperture: below that the 8 fixed samples
///   of the sweep (BLOCK C/D/E of `perception.wgsl`) get squeezed into a <7.5° slice in
///   front of the car — it loses peripheral vision that BUG 1 (historical, see the
///   `LAT_SAMPLES` comment in the shader) already showed to be costly. Ceiling 170°: above that the
///   ½-angle exceeds 85°, almost looking backwards; the sweep has 8 FIXED angles (not
///   adaptive), so it would spend half its resolution behind the car, where no
///   output channel uses the data.
/// - **wall_range_m** (20..120 m): floor 20 m ≈ 0.71 s of anticipation at 28 m/s
///   (maximum observed speed — the same number that already justifies `LAT_SAMPLES` in this
///   file), below which the wall sweep reacts too late for the body (physics
///   in `physics_a.wgsl`) to brake in time. Ceiling 120 m: same order of magnitude as the
///   neighboring block in the SAME kernel (road_ahead FAR already covers 45..150 m) — a wall_range much
///   larger than road_ahead would make the sensor "see" a wall before any corresponding
///   track channel, producing an alarm with no context.
/// - **lat_range_m** (15..100 m): floor 15 m ≈ 0.54 s at 28 m/s — same logic as BUG 1
///   already documented in the shader (27 m / 0.96 s was measured "marginal"; 15 m sits below only
///   because the lateral sweep has 10 dense samples near the body, unlike the
///   frontal one, which has only 1 channel). Ceiling 100 m: same order of magnitude as wall_range,
///   without exceeding a small fraction of the grid (320 cells × 3 m = 960 m; 100 m is
///   <11% of that — it remains a "local" sensor, not a whole-world radar).
/// - **curv_near/mid/far_m** (15..80 / 40..150 / 80..250 m): the three curvature
///   lookaheads of BLOCK K, today fixed at 40/70/120 m. The ranges give ~2× slack on
///   each side of the historical value (which sits near the middle of each range, not at the edge),
///   consistent with the curve-detection ceiling already used in the rest of the file
///   (`perception_temporal.wgsl` probes curvature up to 350 m). We do NOT enforce
///   near < mid < far after mutation: the names are only the BIRTH intent — the
///   `curv_at_d` function (WGSL equivalent) does not distinguish "which" distance is which, it only reads
///   the received value, so evolution can freely redistribute the three.
pub const SENSOR_RANGES: [(f32, f32); N_SENSOR_GENES] = [
    // 30°..85° in radians. The lower bound is exactly FRAC_PI_6, so use the constant
    // rather than a rounded literal; the upper bound has no named constant.
    (std::f32::consts::FRAC_PI_6, 1.483529864), // fov_half_rad
    (20.0, 120.0),                              // wall_range_m
    (15.0, 100.0),                              // lat_range_m
    (15.0, 80.0),                               // curv_near_m
    (40.0, 150.0),                              // curv_mid_m
    (80.0, 250.0),                              // curv_far_m
];

/// Fraction of each gene's RANGE used as the σ of the gaussian mutation (see `mutate_sensors`).
/// 8%: fast enough to explore the whole range in dozens of generations — same
/// order of magnitude as the choice already made for `slow`/`tau` above (few genes against
/// tens of thousands of weights, so higher rate and magnitude than L1/L2).
pub const SENSOR_MUTATE_FRAC: f32 = 0.08;

/// Turns evolution of the sensor bank on/off (SENSORS PHASE, step 1: FOV of the wall/
/// curvature sweep, wall range, lateral range, 3 curvature lookaheads —
/// see `N_SENSOR_GENES`).
///
/// Default `false` (OFF). Rule from this project's CLAUDE.md: *"every change behind
/// a flag defaulting to today's behavior, so as not to invalidate a battery in progress"* —
/// and unlike `SPECTRAL_INIT_ON` (which already had 4 runs measuring the ~7.0 gain
/// before earning default `true`), this change has NO prior measurement. So:
///
/// - Off (default): `Genome::random` always draws `sensors = SENSOR_DEFAULTS`
///   (every genome is born identical — today's hardcodes) and `evolve()` NEVER mutates the
///   field. The behavior of any battery that does not pass a new flag is
///   bit-identical to before this change existed.
/// - On (`--sensores-evolutivos`): `Genome::random` draws within
///   `SENSOR_RANGES` (heterogeneous from birth — different from the τ precedent,
///   which only varies via opt-in `randomize_tau`; here the task itself asked for a direct draw
///   in `Genome::random`) and `evolve()` mutates the field every generation.
///
/// `--sensores-fixos` forces it back to `false` — it is the literal name requested by the
/// task. Even though it is already the default today, existing as a flag allows documenting the
/// intent explicitly in a command (and it wins if it appears AFTER
/// `--sensores-evolutivos` on the command line), the same way `--sem-lento` and
/// `--sem-init-espectral` are the explicit control arms of other experiments.
///
/// # What does NOT inherit from this field (CPU/GPU parity documented, not hidden)
///
/// `apply_to_engine`/`from_engine` (the CPU/Bevy path, via `SovereignEngine`) do not
/// touch `sensors` — `motor.rs` (Bevy) never read `perception.wgsl`, only the training
/// binary (`gpu_evolve`) and `gpu_validate` read it. A champion evolved with heterogeneous
/// sensors and then loaded in the Bevy visualizer will SEE with the usual hardcoded
/// FOV/ranges — a known divergence, same category already accepted for
/// `out_gain`/P1.5 (`docs/PLANO_EVOLUCAO.md`), not a silent bug.
pub static SENSORS_EVOLVE_ON: AtomicBool = AtomicBool::new(false);

/// Minimal abstraction over the two numbers every genome mutation in this file
/// consumes: "did the draw fall within the mutation rate?" (`chance`, uniform `[0,1)`) and "how much
/// to perturb" (`gauss`, N(0,1)).
///
/// Exists only so `mutate_sensors` can receive ONE generator instead of two `FnMut`
/// closures closing over the same variable — `evolve()`'s local `Rng` is `&mut`
/// borrowed by EACH closure, and two live `FnMut`s over the same
/// variable are two simultaneous mutable borrows, which the borrow checker rejects.
/// A `&mut dyn GenoRng` reference borrows only once.
pub trait GenoRng {
    fn chance(&mut self) -> f32;
    fn gauss(&mut self) -> f32;
}

impl GenoRng for crate::rng::Lcg {
    fn chance(&mut self) -> f32 {
        self.uniform()
    }
    fn gauss(&mut self) -> f32 {
        self.gaussian()
    }
}

/// Draws the `N_SENSOR_GENES` genes uniformly within `SENSOR_RANGES` — the
/// HETEROGENEOUS condition (used only when `SENSORS_EVOLVE_ON` is on).
///
/// PURE function (takes the seed, reads no atomic) so it can be tested without the
/// gap 8/R8 concern of this project's CLAUDE.md: toggling a global atomic inside
/// a `#[test]` risks interfering with another test running in parallel in the same
/// process (same reason `spectral_rescale_factor` was extracted as a
/// pure function instead of being tested via `SPECTRAL_INIT_ON`).
pub fn random_sensors_within_ranges(seed: u64) -> Vec<f32> {
    let mut rng = crate::rng::Lcg::new(seed);
    SENSOR_RANGES
        .iter()
        .map(|&(lo, hi)| lo + rng.uniform() * (hi - lo))
        .collect()
}

/// Mutates the sensor bank IN-PLACE — gaussian noise PROPORTIONAL TO EACH GENE'S
/// RANGE (`SENSOR_MUTATE_FRAC × (hi - lo)`), never a fixed absolute σ over all of them.
///
/// Why: `fov_half_rad` is in radians (range ~1.0 wide) and the ranges are in
/// meters (ranges 65 to 170 wide) — a fixed σ in meters would trample the FOV (the
/// angle's entire range would fit inside ONE typical perturbation) or, if calibrated
/// for the FOV, would leave the ranges nearly frozen (the same perturbation in radians is
/// noise in meters). It is exactly gap 11 of the project's CLAUDE.md ("forgotten
/// normalization turning an experiment into an artifact"), now between angle and distance
/// instead of short/long τ — the same problem, solved with the same idea: scale by each
/// channel's own magnitude before comparing.
///
/// Clamp to `SENSOR_RANGES` after each mutation — it never produces a physically
/// impossible sensor (negative FOV, range of zero or less).
///
/// PURE function (receives the generators from outside, does not read `SENSORS_EVOLVE_ON`) for the same
/// reason as `random_sensors_within_ranges` — see the comment there.
pub fn mutate_sensors(sensors: &mut [f32], p_mutate: f32, rng: &mut impl GenoRng) {
    debug_assert_eq!(
        sensors.len(),
        N_SENSOR_GENES,
        "sensor bank must have N_SENSOR_GENES genes"
    );
    for i in 0..N_SENSOR_GENES {
        if rng.chance() < p_mutate {
            let (lo, hi) = SENSOR_RANGES[i];
            let sigma = (hi - lo) * SENSOR_MUTATE_FRAC;
            sensors[i] = (sensors[i] + rng.gauss() * sigma).clamp(lo, hi);
        }
    }
}

/// CfC channels — the filter covers the first 96 of the sensory wave.
pub const N_CFC_CHANNELS: usize = 96;

impl Genome {
    /// Zeroes the L1 column of the channels in `perception::channels::DEAD_L1_INPUTS`
    /// (today only TireTemp, ch70 — structurally constant, see the full doc in
    /// `Genome::random`). Extracted into a method because `Genome::random` zeroed only
    /// at BIRTH — `evolve()` mutates L1 every generation without reapplying the mask, and
    /// measured on the real champion (`evolve_best_gen0059.bin`, ch70 column) the 96
    /// weights were already 100% non-zero (mean |w| 0.2085, equal to the magnitude of the
    /// live neighboring channel ch71) — the protection erodes generation by generation (finding
    /// P1.8 "DEAD_L1_INPUTS zeroes only at birth").
    ///
    /// Call after any L1 mutation. Do NOT call in `load_genome` —
    /// that would silently rewrite real weights a champion has already trained
    /// (same principle as the test `genoma_legado_carregado_do_disco_nao_e_reescalado`:
    /// loading from disk does not modify weights).
    pub fn zero_dead_inputs(&mut self) {
        for &(ch, _motivo) in crate::perception::channels::DEAD_L1_INPUTS {
            for neuron in 0..L1_NEURONS {
                self.l1[neuron * SENSORY_WAVE_SIZE + ch] = 0.0;
            }
        }
    }

    /// Generates a genome with random weights (N(0, σ_l1)).
    ///
    /// σ_l1 = 0.30 for L1/L2 — placed at the scale of real weights observed
    /// in CHAMPION.dna (mean |w| ≈ 0.15, max ≈ 0.7). Zeroed weights = null network.
    /// σ = 0.3 guarantees at least ~30% of weights non-null with non-trivial action.
    ///
    /// After drawing L1, and if [`SPECTRAL_INIT_ON`] is on (default), L1 is
    /// rescaled to be born with spectral gain ≈ `thresholds::SPECTRAL_INIT_TARGET`
    /// — see the "Spectral initialization of L1 (P0.5)" module section at the top of the file.
    ///
    /// KAN knots = default (tanh ramp) — works as a smooth identity and does not
    /// need to be evolved in the early stages.
    pub fn random(seed: u64) -> Self {
        // 64-bit LCG (Knuth MMIX) — deterministic per seed.
        let mut state = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0x6C62272E07BB0142);
        let mut rng_f32 = move || -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f32) / (1u64 << 53) as f32;
            u * 2.0 - 1.0 // uniform [-1, 1]
        };

        // Box-Muller for N(0, σ).
        let sigma_l1 = 0.30_f32;
        let gauss = |rng: &mut dyn FnMut() -> f32| -> f32 {
            let u1 = (rng() * 0.5 + 0.5).max(1e-7_f32);
            let u2 = rng() * 0.5 + 0.5;
            let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            (g * sigma_l1).clamp(-3.0 * sigma_l1, 3.0 * sigma_l1) // clamp at 3σ
        };

        let mut l1: Vec<f32> = (0..N_L1_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();
        let l2: Vec<f32> = (0..N_L2_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();

        // P0.5 — spectral initialization: measures the gain the draw above just
        // produced (same computation `analysis::activation` uses for the alarm,
        // `SpectralRadiusHigh`) and rescales L1 by a single scalar factor so the
        // measured gain hits `thresholds::SPECTRAL_INIT_TARGET`. A scalar factor
        // over the whole matrix scales the largest singular value by the same factor
        // (linear — see the test in `math/linalg.rs`), so the target is hit
        // exactly, not by trial. It only touches L1: it is the only matrix the alarm
        // audits (L2 uses `w_rec: None` in the `collect_activation_stats` call).
        //
        // `SPECTRAL_INIT_ON` is the A/B control arm (`--sem-init-espectral`,
        // default ON). A genome coming from disk (`load_genome`) never passes through here.
        if SPECTRAL_INIT_ON.load(Ordering::Relaxed) {
            let medido = linalg::largest_singular_value(&l1, L1_NEURONS, SENSORY_WAVE_SIZE);
            let fator = spectral_rescale_factor(medido, thresholds::SPECTRAL_INIT_TARGET);
            for w in l1.iter_mut() {
                *w *= fator;
            }
        }

        // CHANNELS PHASE (2026-08-14) — zeroes the L1 column of the channels in
        // `perception::channels::DEAD_L1_INPUTS` (today only TireTemp, ch70): channels
        // whose kernel writes but whose value is structurally constant (no real data
        // behind it — see justification in the record). The brain is not born spending
        // input weight on static. It runs AFTER the spectral rescale on
        // purpose: measure/normalize the gain of the whole i.i.d. matrix first, and only
        // then zero the ~96 inputs (out of 18.432) of the dead channels — a perturbation
        // small enough (<1% of the weights) not to push the measured gain out of
        // [0.8, 1.2] (see test `genoma_novo_nasce_com_ganho_espectral_perto_de_1`).
        // This is NOT neuron pruning — the 96→64 ablation (PLANO_EVOLUCAO.md) proved that
        // no L1 neuron is disposable; this only prevents an INPUT weight from
        // reading a channel that never varies.
        for &(ch, _motivo) in crate::perception::channels::DEAD_L1_INPUTS {
            for neuron in 0..L1_NEURONS {
                l1[neuron * SENSORY_WAVE_SIZE + ch] = 0.0;
            }
        }
        // (it does not call `Genome::zero_dead_inputs` here because `self` does not exist yet
        // — `l1` is a local Vec that only becomes a Genome field in the `Self { .. }` below.
        // Same logic, see the method's doc for why `evolve()` MUST
        // reapply this and `load_genome` must not.)

        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();

        // Slow brain: starts from the default initialization and diverges via mutation. It is born
        // identical in all beings on purpose — the difference has to be SELECTED,
        // not drawn, otherwise the first generation already mixes two effects.
        let slow = crate::engine::SlowBrain::new().all_weights();
        // Uniform τ = inherited behavior. Heterogeneous only via `randomize_tau`, so that
        // the experiment's control arm is the old brain, not a third case.
        let tau = vec![crate::cfc::DECAY_DEFAULT; N_CFC_CHANNELS];
        // SENSORS PHASE (step 1): off (default), sensors = SENSOR_DEFAULTS always —
        // today's hardcodes, identical for the whole population. On
        // (`SENSORS_EVOLVE_ON`), draws within `SENSOR_RANGES` — heterogeneous from
        // birth (unlike τ: the task itself asked for a direct draw here, not a
        // separate opt-in `randomize_sensors`). Seed with its own offset so as not to repeat the
        // sequence that already generated l1/l2/tau for the same `seed`.
        let sensors = if SENSORS_EVOLVE_ON.load(Ordering::Relaxed) {
            random_sensors_within_ranges(seed.wrapping_add(0x5E45_0000_5E45_0000))
        } else {
            SENSOR_DEFAULTS.to_vec()
        };
        // P1.7 — the three mechanisms ALWAYS are born neutral, no atomic gate (unlike
        // `sensors`/`tau`): there is no request for "draw heterogeneous
        // from birth" for these fields — only mutation, generation after generation,
        // diverging from neutral slowly. See the field docs in `Genome` for the
        // math of why each default reproduces today's brain exactly.
        let neuron_alpha = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        let neuron_period_l1 = vec![1.0; N_L1_NEURONS];
        let neuron_threshold_l1 = vec![0.0; N_L1_NEURONS];
        let neuron_activation = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        Self {
            l1,
            l2,
            kan_knots,
            slow,
            tau,
            sensors,
            neuron_alpha,
            neuron_period_l1,
            neuron_threshold_l1,
            neuron_activation,
            gates_w: vec![0.0; 16],
        }
    }

    /// Draws the τ bank log-uniform in [25 ms, 4 s] — the heterogeneous condition.
    pub fn randomize_tau(&mut self, seed: u64) {
        self.tau = crate::cfc::log_uniform_decays(seed, N_CFC_CHANNELS);
    }

    /// Flattens the τ bank of several genomes into the layout `BrainGpu::upload_tau` expects.
    pub fn pack_taus(genomes: &[Genome]) -> Vec<f32> {
        let mut out = Vec::with_capacity(genomes.len() * N_CFC_CHANNELS);
        for g in genomes {
            out.extend_from_slice(&g.tau);
        }
        out
    }

    /// Flattens the sensor bank of several genomes into the layout
    /// `PerceptionGpu::upload_sensors` expects — same pattern as `pack_taus`.
    pub fn pack_sensors(genomes: &[Genome]) -> Vec<f32> {
        let mut out = Vec::with_capacity(genomes.len() * N_SENSOR_GENES);
        for g in genomes {
            out.extend_from_slice(&g.sensors);
        }
        out
    }

    /// P1.7 — flattens the per-neuron metadata (mechanisms 1/2/3) of several
    /// genomes into the SoA layout `BrainGpu::upload_neuron_meta` expects — same
    /// pattern as `pack_taus`/`pack_sensors`. Per genome, 480 floats in the SAME
    /// order as `NEURON_META_PER_SER` in `brain.wgsl`:
    ///   `l1_alpha(96) ++ l1_period(96) ++ l1_threshold(96) ++ l1_activation(96)
    ///    ++ l2_alpha(48) ++ l2_activation(48)`.
    /// `neuron_alpha`/`neuron_activation` hold L1+L2 together (`[0..96)`=L1,
    /// `[96..144)`=L2, see the field docs) — this function is the one that separates the two when
    /// building the buffer, because the shader wants the blocks NOT interleaved
    /// (all l1_* before all l2_*, so the `NEURON_META_L1_*_OFF` offsets
    /// can be fixed constants instead of depending on `n_beings`).
    pub fn pack_neuron_meta(genomes: &[Genome]) -> Vec<f32> {
        let per_ser = N_L1_NEURONS * 4 + N_L2_NEURONS * 2; // 480, mirrors NEURON_META_PER_SER
        let mut out = Vec::with_capacity(genomes.len() * per_ser);
        for g in genomes {
            debug_assert_eq!(g.neuron_alpha.len(), N_L1_NEURONS + N_L2_NEURONS);
            debug_assert_eq!(g.neuron_period_l1.len(), N_L1_NEURONS);
            debug_assert_eq!(g.neuron_threshold_l1.len(), N_L1_NEURONS);
            debug_assert_eq!(g.neuron_activation.len(), N_L1_NEURONS + N_L2_NEURONS);
            out.extend_from_slice(&g.neuron_alpha[..N_L1_NEURONS]); // l1_alpha
            out.extend_from_slice(&g.neuron_period_l1); // l1_period
            out.extend_from_slice(&g.neuron_threshold_l1); // l1_threshold
            out.extend_from_slice(&g.neuron_activation[..N_L1_NEURONS]); // l1_activation
            out.extend_from_slice(&g.neuron_alpha[N_L1_NEURONS..]); // l2_alpha
            out.extend_from_slice(&g.neuron_activation[N_L1_NEURONS..]); // l2_activation
        }
        out
    }

    /// Loads a genome from an existing `SovereignEngine`.
    pub fn from_engine(eng: &SovereignEngine) -> Self {
        let l1 = eng.instinct_layer.weights.clone();
        let l2 = eng.cortex_l2.weights.clone();
        // KAN knots reside in engine.kan_filters (per-channel), but the OUTPUT
        // knots (throttle/steer/brake) are not in the engine — we use the default.
        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();
        let slow = eng.slow_brain.all_weights();
        let tau = vec![crate::cfc::DECAY_DEFAULT; N_CFC_CHANNELS];
        // `SovereignEngine` (CPU/Bevy path) has no concept of heritable sensors —
        // see "What does NOT inherit from this field" in the `SENSORS_EVOLVE_ON` doc.
        let sensors = SENSOR_DEFAULTS.to_vec();
        // P1.7 — same reason as `sensors`: the CPU/Bevy path (`SovereignEngine`)
        // has no concept of an asynchronous brain, only the GPU training binary reads
        // these fields. Born neutral, same as `Genome::random`.
        let neuron_alpha = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        let neuron_period_l1 = vec![1.0; N_L1_NEURONS];
        let neuron_threshold_l1 = vec![0.0; N_L1_NEURONS];
        let neuron_activation = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        Self {
            l1,
            l2,
            kan_knots,
            slow,
            tau,
            sensors,
            neuron_alpha,
            neuron_period_l1,
            neuron_threshold_l1,
            neuron_activation,
            gates_w: vec![0.0; 16],
        }
    }

    /// Writes a genome into a `SovereignEngine` (replaces weights).
    ///
    /// Fails explicitly if L1/L2 diverge — silent truncation already caused
    /// weights lost without warning, result: beings were born with a partially zeroed genome
    /// while selection metrics looked normal.
    /// How many L1 neurons stay ALIVE. `0` = all (normal behavior).
    ///
    /// Leonardo: *"we can evaluate how much space they actually need, and if the
    /// model gets short on space we increase it."*
    ///
    /// The capacity measure says L1 uses 6.7 of its 96 dimensions (7%) — but a metric is not
    /// proof. Zeroing the output weights of the neurons above the cut and measuring the meters
    /// turns the estimate into an experiment: if cutting 96→16 costs no performance, the
    /// 93% idle are demonstrated, not inferred.
    ///
    /// The OUTPUT is zeroed (the L2 column that reads that neuron) and not the input, because that is
    /// what removes its contribution to behavior; zeroing the input would leave
    /// the neuron emitting its bias.
    pub fn ablar_l1_acima_de(&mut self, vivos: usize) {
        use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS};
        if vivos == 0 || vivos >= L1_NEURONS {
            return;
        }
        for j in 0..L2_NEURONS {
            for n in vivos..L1_NEURONS {
                let idx = j * L1_NEURONS + n;
                if idx < self.l2.len() {
                    self.l2[idx] = 0.0;
                }
            }
        }
    }

    pub fn apply_to_engine(&self, eng: &mut SovereignEngine) {
        assert_eq!(
            self.l1.len(),
            eng.instinct_layer.weights.len(),
            "L1 has {} weights, engine expects {} — SENSORY_WAVE_SIZE or L1_NEURONS divergence",
            self.l1.len(),
            eng.instinct_layer.weights.len()
        );
        eng.instinct_layer.weights.copy_from_slice(&self.l1);
        assert_eq!(
            self.l2.len(),
            eng.cortex_l2.weights.len(),
            "L2 has {} weights, engine expects {} — L1_NEURONS or L2_NEURONS divergence",
            self.l2.len(),
            eng.cortex_l2.weights.len()
        );
        eng.cortex_l2.weights.copy_from_slice(&self.l2);
        if !self.slow.is_empty() {
            eng.slow_brain.set_all_weights(&self.slow);
        }
        // KAN knots are used by the evolution cycle directly (not in the engine).
    }
}

/// Evolutionary algorithm parameters.
pub struct EvoParams {
    /// Per-weight mutation rate: probability of each weight being perturbed.
    pub p_mutate: f32,
    /// Standard deviation of the gaussian noise added to each mutated weight.
    pub sigma: f32,
    /// Number of elite genomes that pass through without mutation.
    pub n_elite: usize,
    /// Tournament size for parent selection.
    pub tournament_k: usize,
    /// Freezes mutation of the BLIND L2 rows (5..48). Investigation 2026-08-14:
    /// no pipeline reads final_intent[5..48) — that is 4.128 weights of inert
    /// search space, carried by genetic hitchhiking through elitism (blind rows were
    /// statistically indistinguishable from the read ones in cosine/kurtosis across the 2 seeds).
    /// Rows 0..l2_linhas_lidas remain mutable.
    pub congela_l2_cegas: bool,
    /// How many L2 rows have a real consumer IN THE CURRENT MODE: 3 (throttle/steer/
    /// brake) with gearbox locked/auto, 5 with `--cambio neural` (intent[3]/[4]).
    /// Finding from the P1.9 battery: freezing with the fixed value 5 left rows 3-4
    /// mutable even with no reader in locked mode (default).
    pub l2_linhas_lidas: usize,
    /// M3: reimposes the region mask after each mutation (see mascara_permite).
    pub mascara_regioes: bool,
}

/// L2 rows with a consumer in the Bevy pipeline (compat for tests and default).
pub const L2_LINHAS_LIDAS: usize = 5;

/// E-REG (M3, docs/REGIOES_CEREBRAIS.md): DIRECTED region mask — reference ceiling
/// after M1 (cost in life) failed 2× (inter-individual variance
/// of the cost ≈ 0; selection cannot see wiring). 4 blocks of 24 L1 neurons:
///   block 0 (j 0-23):  VISION       — channels 0-15 (retina) + 128-191 (fovea/periphery/horizon)
///   block 1 (j 24-47): WALL/ROAD — channels 16-63
///   block 2 (j 48-71): BODY/PROX  — channels 64-95
///   block 3 (j 72-95): ASSOCIATIVE — all 192 (integrator hub)
/// Weight outside the mask = 0, reimposed post-mutation and on load (same machinery
/// as the old DEAD_L1_INPUTS, now per region).
pub fn mascara_permite(j: usize, c: usize) -> bool {
    match j / 24 {
        0 => c < 16 || c >= 128,
        1 => (16..64).contains(&c),
        2 => (64..96).contains(&c),
        _ => true,
    }
}

/// Zeroes the weights outside the region mask (call post-mutation and post-load).
pub fn aplicar_mascara_regioes(g: &mut Genome) {
    for j in 0..L1_NEURONS {
        for c in 0..SENSORY_WAVE_SIZE {
            if !mascara_permite(j, c) {
                g.l1[j * SENSORY_WAVE_SIZE + c] = 0.0;
            }
        }
    }
}

impl Default for EvoParams {
    fn default() -> Self {
        Self {
            p_mutate: 0.02,  // 2% of weights per generation
            sigma: 0.05,     // gaussian perturbation σ=0.05
            n_elite: 3,      // top 3 pass without mutation
            tournament_k: 3, // tournament of 3
            congela_l2_cegas: false,
            l2_linhas_lidas: L2_LINHAS_LIDAS,
            mascara_regioes: false,
        }
    }
}

/// Result of one generation.
pub struct GenResult {
    pub generation: usize,
    pub dist_max: f64,  // meters, best individual
    pub dist_med: f64,  // meters, median
    pub n_ge1v: usize,  // beings with ≥1 complete lap
    pub n_ge2v: usize,  // beings with ≥2 complete laps
    pub mortality: f32, // fraction of beings that did not move (dist < 10m)
}

/// Executes one evolutionary cycle: selects + mutates.
///
/// `fitnesses[i]` = accumulated distance in meters of being i in the previous generation.
/// Returns the new generation of genomes in the same order (0..pop_size).
///
/// `pop_seed` (the CLI's `--seed`) ENTERS the mutation seed. Previously it did
/// not: the seed was pure f(generation), and under `--retomar` (population
/// derived from the champion, no `Genome::random`) two executions with `--seed 77` and
/// `--seed 31` produced byte-identical genomes in every generation (measured via
/// md5sum in pos-auditoria-s{77,31}, gens 0/5/10/15/18/19). `pop_seed=0`
/// reproduces the old stream exactly (the term cancels out) — old batteries
/// remain reproducible.
pub fn evolve(
    genomes: &[Genome],
    fitnesses: &[f64],
    params: &EvoParams,
    generation: usize, // generation index (for deterministic seed)
    pop_seed: u64,     // the CLI's `--seed`; 0 = historical stream
) -> Vec<Genome> {
    let n = genomes.len();
    assert_eq!(
        fitnesses.len(),
        n,
        "fitnesses and genomes must have the same size"
    );

    // ── Sort by descending fitness ─────────────────────────────────────────
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        fitnesses[b]
            .partial_cmp(&fitnesses[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut nova: Vec<Genome> = Vec::with_capacity(n);

    // ── Elitism: the best pass through unchanged ────────────────────────────────
    for &idx in order.iter().take(params.n_elite) {
        nova.push(genomes[idx].clone());
    }

    // ── Tournament selection + mutation ───────────────────────────────────────
    // Local struct to encapsulate the RNG without closure aliasing.
    struct Rng {
        state: u64,
        spare: Option<f32>,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                spare: None,
            }
        }
        fn u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }
        fn f32(&mut self) -> f32 {
            ((self.u64() >> 11) as f32) / (1u64 << 53) as f32
        }
        // Box-Muller N(0,1).
        fn gauss(&mut self) -> f32 {
            if let Some(v) = self.spare.take() {
                return v;
            }
            // No rejection loop here: `u1` is clamped away from zero, so the first
            // draw is always valid. The `loop` this used to sit in never iterated
            // twice and clippy reported it as dead structure.
            let u1 = self.f32().max(1e-7_f32);
            let u2 = self.f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            self.spare = Some(r * theta.sin());
            r * theta.cos()
        }
    }
    // `Rng` implements `GenoRng` so `mutate_sensors` can receive ONE reference
    // (`&mut rng`) instead of two closures — see the `GenoRng` doc at the top of the file.
    impl GenoRng for Rng {
        fn chance(&mut self) -> f32 {
            self.f32()
        }
        fn gauss(&mut self) -> f32 {
            Rng::gauss(self)
        }
    }

    let seed = (generation as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xCAFE_BABE_DEAD_BEEF)
        .wrapping_add(pop_seed.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let mut rng = Rng::new(seed);

    while nova.len() < n {
        // Tournament: selects the best among k random candidates.
        let parent_idx = {
            let mut best = rng.u64() as usize % n;
            for _ in 1..params.tournament_k {
                let cand = rng.u64() as usize % n;
                if fitnesses[cand] > fitnesses[best] {
                    best = cand;
                }
            }
            best
        };

        let pai = &genomes[parent_idx];
        let mut filho = pai.clone();

        // Gaussian mutation on L1 (2% of weights per generation, configurable σ).
        for w in filho.l1.iter_mut() {
            if rng.f32() < params.p_mutate {
                *w += rng.gauss() * params.sigma;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // P1.8, finding "DEAD_L1_INPUTS zeroes only at birth" — reapplies the mask
        // of the dead channels (ch70/TireTemp) after the L1 mutation, so it does not
        // erode generation by generation (measured on the real champion: 96/96 weights already
        // resurrected). Safe to call here: `evolve()` is what PRODUCES new genomes
        // each generation, so this is part of the normal process of generating
        // children — no test assumes `evolve()` preserves the raw mutation value
        // on those channels.
        filho.zero_dead_inputs();
        if params.mascara_regioes {
            aplicar_mascara_regioes(&mut filho);
        }
        // Gaussian mutation on L2. With `congela_l2_cegas`, only the rows with a
        // real consumer (0..L2_LINHAS_LIDAS) mutate — the blind ones stay as they are
        // (not even a random walk: whatever the champion had stays frozen in it).
        let l2_limite = if params.congela_l2_cegas {
            params.l2_linhas_lidas.min(L2_NEURONS) * L1_NEURONS
        } else {
            filho.l2.len()
        };
        for w in filho.l2[..l2_limite].iter_mut() {
            if rng.f32() < params.p_mutate {
                *w += rng.gauss() * params.sigma;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // Slow-brain mutation. Larger σ because it is a thousand weights against the fast
        // brain's twenty-six thousand — with the same rate the slow brain would receive practically null perturbation.
        // M4: the conductor's gates mutate like the slow brain (few weights → slow brain's σ).
        for w in filho.gates_w.iter_mut() {
            if rng.f32() < params.p_mutate * 4.0 {
                *w += rng.gauss() * params.sigma * 2.0;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        for w in filho.slow.iter_mut() {
            if rng.f32() < params.p_mutate * 3.0 {
                *w += rng.gauss() * params.sigma * 1.5;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // τ bank mutation: MULTIPLICATIVE, in log.
        //
        // Additive would be wrong here. The range goes from 0.25 to 40 s⁻¹ — a σ that truly
        // moves a channel at 0.25 tramples one at 40, and vice versa. What has constant
        // scale is the ratio: ×1.15 means the same thing anywhere in the range.
        //
        // Rate 3× that of the weights for the same reason as the slow brain: 96 values against
        // 26 thousand, and at the base rate the τ bank would barely be touched in a lifetime of 30 generations.
        let d_lo = 1.0 / crate::cfc::TAU_MAX_S;
        let d_hi = 1.0 / crate::cfc::TAU_MIN_S;
        for d in filho.tau.iter_mut() {
            if rng.f32() < params.p_mutate * 3.0 {
                *d = (*d * (rng.gauss() * 0.20).exp()).clamp(d_lo, d_hi);
            }
        }
        // Sensor bank mutation (SENSORS PHASE) — only when sensor evolution is
        // on (`SENSORS_EVOLVE_ON`, default OFF). Off, the
        // field stays bit-identical to `SENSOR_DEFAULTS` for the battery's entire life — see the
        // `SENSORS_EVOLVE_ON` doc for why this gate is stronger than τ's
        // (τ mutates unconditionally; sensors only mutate with explicit confirmation).
        // Same 3× rate as slow/τ for the same reason: 6 genes against tens of thousands
        // of weights — the base rate would barely touch the bank over an entire lifetime of generations.
        if SENSORS_EVOLVE_ON.load(Ordering::Relaxed) {
            mutate_sensors(&mut filho.sensors, params.p_mutate * 3.0, &mut rng);
        }
        // kan_knots (OUTPUT, 3×12): NOT mutated — finding P1.8 "kan_knots is mutated
        // every generation but never read back". `main.rs` passes `&kan_knots_global`
        // (`default_kan_knots()`, fixed) to `ctx.tick()` at ALL real call sites
        // (main loop and held-out) — `genomes[i].kan_knots` is never read outside
        // save/load. Mutating here was RNG/CPU wasted per individual/generation, AND
        // it polluted the `.dna` with "evolution" that does not exist (the field changes on disk, the
        // behavior does not). If/when someone rewires per-being knots in
        // `ctx.tick()` (see the field's doc, full finding in docs/PLANO_EVOLUCAO.md
        // P1.8), the mutation comes back here.

        // ── P1.7 — asynchronous brain (docs/PLANO_EVOLUCAO.md §P1.7) ────────────
        // The four blocks below are ALWAYS the LAST to consume `rng` in this
        // loop — l1/l2/slow/tau/sensors (above) mutate exactly as they mutated
        // before P1.7 existed, byte for byte, because none of them had even a
        // single `rng` call inserted BEFORE them. A genome that is born neutral (every
        // `Genome::random`) and never mutates these fields (the four atomics
        // turned off via `--sem-*`) reproduces today's output generation after generation
        // — it is the strongest control arm one can write. (`kan_knots`
        // stopped mutating in P1.8 — the guarantee above is about l1/l2/slow/tau/sensors,
        // it does not change with that removal.)

        // Mechanism 1 (memory): alpha mutates ADDITIVE-clamped, not multiplicative
        // like τ — τ is a decay rate in s⁻¹ (a quantity with its own scale,
        // hence log-uniform); alpha is a mixing weight already normalized in
        // [0,1), so a small gaussian step is the direct choice, same
        // spirit as the rest of the "few genes against tens of thousands of
        // weights" fields (3× rate, σ larger than the weights').
        if NEURON_MEM_MUTATE_ON.load(Ordering::Relaxed) {
            for a in filho.neuron_alpha.iter_mut() {
                if rng.f32() < params.p_mutate * 3.0 {
                    *a = (*a + rng.gauss() * 0.10).clamp(0.0, 0.95);
                }
            }
        }
        // Mechanism 2, period: INTEGER — discrete ±1 step, not continuous
        // gaussian (1.3→1.31 does not change the neuron's behavior in any tick;
        // 1→2 does). Ceiling 8: above that the neuron is already almost always "mute"
        // (keeping the old value 7 out of every 8 ticks) with no additional representation
        // gain — same physical-range logic as the rest of the file, except
        // here the "physical" is "still reacts fast enough for the track".
        if NEURON_PERIOD_MUTATE_ON.load(Ordering::Relaxed) {
            for p in filho.neuron_period_l1.iter_mut() {
                if rng.f32() < params.p_mutate * 3.0 {
                    let passo = if rng.f32() < 0.5 { -1.0 } else { 1.0 };
                    *p = (*p + passo).round().clamp(1.0, 8.0);
                }
            }
        }
        // Mechanism 2, threshold: gaussian-clamped, same recipe as alpha. Ceiling
        // 0.9: above that the neuron would rarely cross the threshold (the
        // post-activation range is [-1,1] or [0,∞) for ReLU/linear, but the
        // quantization clamp in brain.wgsl already saturates at [-1,1] before the ×127) and would stay
        // permanently mute — that is not "asynchrony", it is switching the neuron off
        // entirely, which already exists as a separate path (L1 ablation).
        if NEURON_THRESHOLD_MUTATE_ON.load(Ordering::Relaxed) {
            for t in filho.neuron_threshold_l1.iter_mut() {
                if rng.f32() < params.p_mutate * 3.0 {
                    *t = (*t + rng.gauss() * 0.05).clamp(0.0, 0.9);
                }
            }
        }
        // Mechanism 3 (heterogeneous activation): DISCRETE mutation — category
        // swap, not perturbation. Rate LOWER than the weights' gaussian (and
        // lower than the 3× rate used above): swapping tanh→ReLU changes the neuron's
        // entire REGIME (from smooth saturation, always active, to
        // hard sparsity, half at exact zero) — it is not a fine adjustment like
        // moving a weight by σ=0.05, it is a discrete decision whose effect only
        // selection measures after several generations. A high rate would make the
        // activation repertoire oscillate generation to generation without time for
        // selection to prove whether the swap helped — same family of reasoning as the KAN
        // knots (0.5%) just above, but even more conservative because here the
        // "cost" of being wrong is bigger (it changes the whole signal that reaches L2).
        if NEURON_ACTIVATION_MUTATE_ON.load(Ordering::Relaxed) {
            const ACT_MUTATE_RATE: f32 = 0.002; // 0.2% per neuron per generation
            for act in filho.neuron_activation.iter_mut() {
                if rng.f32() < ACT_MUTATE_RATE {
                    *act = (rng.f32() * 4.0).floor().min(3.0);
                }
            }
        }

        nova.push(filho);
    }

    nova
}

/// P1.7, mechanism 2 — how many L1 neurons (entire population) EFFECTIVELY
/// recompute their memory on each tick, given the period bank already uploaded to the GPU.
///
/// Mirrors EXACTLY the `fires` condition of `brain.wgsl`
/// (`period <= 1 || tick_counter % period == 0`, counter starting at 1 — the
/// shader increments the counter BEFORE any read, `counter_old=0` on the
/// first tick becomes `counter_new=1`) — simulated here in pure Rust, NO GPU,
/// because each neuron's period is known in advance (it does not depend on the
/// sensory data of that tick, only on the tick index). HONESTY (CLAUDE.md/
/// task): this measures REPRESENTATION (how many neurons are "awake"
/// at that instant), not GPU time — all threads run anyway
/// on every dispatch, period>1 does not save a single clock cycle.
///
/// `n_ticks` ideally covers at least one complete LCM cycle of the periods
/// in use (LCM(1..8)=840) so the mean/deviation do not depend on WHERE the window
/// started sampling — the caller passes `n_ticks.min(840)` or more.
///
/// Returns `(media, desvio_padrao)` (mean, standard deviation) of the count of neurons firing per
/// tick, summed over ALL beings (not per being).
pub fn simulate_l1_fire_counts(periods_all: &[f32], n_beings: usize, n_ticks: usize) -> (f64, f64) {
    if n_beings == 0 || n_ticks == 0 {
        return (0.0, 0.0);
    }
    debug_assert_eq!(
        periods_all.len(),
        n_beings * N_L1_NEURONS,
        "periods_all should have n_beings*N_L1_NEURONS values"
    );
    let mut contagens: Vec<f64> = Vec::with_capacity(n_ticks);
    for tick in 1..=n_ticks {
        let mut count: u64 = 0;
        for &p_raw in periods_all.iter() {
            let period = (p_raw.round() as i64).max(1) as usize;
            if period <= 1 || tick % period == 0 {
                count += 1;
            }
        }
        contagens.push(count as f64);
    }
    let media = contagens.iter().sum::<f64>() / n_ticks as f64;
    let var = contagens.iter().map(|&c| (c - media).powi(2)).sum::<f64>() / n_ticks as f64;
    (media, var.sqrt())
}

/// Computes a generation's statistics from the accumulated distances.
pub fn gen_stats(generation: usize, dist_acum: &[f64], lap_dist: f64) -> GenResult {
    let n = dist_acum.len();
    let dist_max = dist_acum.iter().cloned().fold(0.0_f64, f64::max);
    let dist_med = {
        let mut v = dist_acum.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let n_ge1v = dist_acum.iter().filter(|&&d| d >= lap_dist).count();
    let n_ge2v = dist_acum.iter().filter(|&&d| d >= lap_dist * 2.0).count();
    let n_morto = dist_acum.iter().filter(|&&d| d < 10.0).count();
    let mortality = n_morto as f32 / n as f32;
    GenResult {
        generation,
        dist_max,
        dist_med,
        n_ge1v,
        n_ge2v,
        mortality,
    }
}

/// Computes the linear slope (OLS) and its standard deviation (estimator error).
///
/// `xs[i]` = generation i (float), `ys[i]` = median of generation i.
/// Returns `(slope, slope_stderr)`. If n < 2 returns (0.0, 0.0).
pub fn linear_slope(ys: &[f64]) -> (f64, f64) {
    let n = ys.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let sxx: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
        .sum();
    if sxx < 1e-10 {
        return (0.0, 0.0);
    }
    let slope = sxy / sxx;
    // SSR = residual variance
    let ssr: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (y - (mean_y + slope * (x - mean_x))).powi(2))
        .sum();
    let var_resid = ssr / (n as f64 - 2.0).max(1.0);
    let slope_stderr = (var_resid / sxx).sqrt();
    (slope, slope_stderr)
}

// ---------------------------------------------------------------------------
// Unit tests — P0.5 (L1 spectral initialization).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `spectral_rescale_factor` is a pure mathematical function — the central test of the
    /// spectral initialization that does not need to touch `SPECTRAL_INIT_ON` nor draw
    /// weights: given the ~7.0 gain measured empirically (project CLAUDE.md, 4 independent
    /// runs), the returned factor has to lead exactly to 1.0.
    #[test]
    fn spectral_rescale_factor_atinge_alvo_exatamente() {
        let fator = spectral_rescale_factor(7.0, 1.0);
        assert!(
            (fator - 1.0 / 7.0).abs() < 1e-6,
            "factor = {fator}, expected 1/7"
        );
        assert!(
            (7.0 * fator - 1.0).abs() < 1e-6,
            "applying the factor should lead to 1.0"
        );
    }

    /// Gain ~0 (null L1, never observed in practice) must not produce an explosive division
    /// — the factor falls back to 1.0 (identity: nothing to scale).
    #[test]
    fn spectral_rescale_factor_ganho_quase_zero_nao_explode() {
        assert_eq!(spectral_rescale_factor(0.0, 1.0), 1.0);
        assert_eq!(spectral_rescale_factor(1e-9, 1.0), 1.0);
    }

    /// P0.5, requirement 4: a new genome (default flag ON) is born with a measured L1
    /// spectral gain in [0.8, 1.2] — not exactly 1.0 because the normalization measures the
    /// GAIN OF THE WHOLE drawn MATRIX (it should hit ~1.0 within the numerical error of the
    /// power-iteration, ~100 iterations, plus the effect of the 3σ clamp on the tail of the
    /// gaussian), but far enough from the uncorrected ~7.0 to prove the
    /// rescale is actually happening.
    ///
    /// Does not touch `SPECTRAL_INIT_ON` (stays at the test process's default `true`) —
    /// see the `spectral_rescale_factor` doc on why to avoid toggling the global
    /// atomic inside tests.
    #[test]
    fn genoma_novo_nasce_com_ganho_espectral_perto_de_1() {
        assert!(
            SPECTRAL_INIT_ON.load(Ordering::Relaxed),
            "test precondition: the flag should be at the default (on) in this process"
        );
        for seed in [1u64, 2, 3, 42, 999_999] {
            let g = Genome::random(seed);
            let ganho = linalg::largest_singular_value(&g.l1, L1_NEURONS, SENSORY_WAVE_SIZE);
            assert!(
                (0.8..=1.2).contains(&ganho),
                "seed {seed}: L1 spectral gain = {ganho}, expected in [0.8, 1.2]"
            );
        }
    }

    /// CHANNELS PHASE (2026-08-14): a new genome is born with the L1 column of the
    /// `DEAD_L1_INPUTS` channels (today only TireTemp, ch70) entirely zeroed — none of the
    /// `L1_NEURONS` weights reading that channel escapes the zeroing, under any seed.
    #[test]
    fn genoma_novo_zera_l1_dos_canais_mortos() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        // Empty list since 2026-08-15 (ch70 refuted by measurement — the channel was
        // ALIVE; thermals implemented in physics_power_c.wgsl). The test becomes a no-op
        // until a genuinely dead channel exists; the machinery stays in the forward path.
        if DEAD_L1_INPUTS.is_empty() {
            return;
        }
        for seed in [1u64, 7, 999] {
            let g = Genome::random(seed);
            for &(ch, _motivo) in DEAD_L1_INPUTS {
                for neuron in 0..L1_NEURONS {
                    let w = g.l1[neuron * SENSORY_WAVE_SIZE + ch];
                    assert_eq!(
                        w, 0.0,
                        "seed {seed}: l1[neuron={neuron}, ch={ch}] = {w}, expected 0.0"
                    );
                }
            }
        }
    }

    /// CHANNELS PHASE: zeroing the dead channels must not change NEIGHBORING channels — sanity
    /// against an index error (off-by-one in the column) zeroing the wrong channel.
    /// Does not touch `SPECTRAL_INIT_ON` (same care as
    /// `genoma_novo_nasce_com_ganho_espectral_perto_de_1` — see the
    /// `spectral_rescale_factor` doc on why to avoid toggling the global atomic
    /// inside tests, R8 of the project's CLAUDE.md). A neighbor ending up at exactly 0.0
    /// after spectral rescaling has practically null probability
    /// (continuous gaussian × non-null scalar factor); multiple seeds reduce
    /// the false-negative risk even further.
    #[test]
    fn zerar_canal_morto_nao_afeta_canal_vizinho_vivo() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        for seed in [1u64, 7, 999] {
            let g = Genome::random(seed);
            for &(ch, _) in DEAD_L1_INPUTS {
                if ch + 1 < SENSORY_WAVE_SIZE {
                    // `0 * SENSORY_WAVE_SIZE` is written out on purpose: it names the
                    // L1 neuron being indexed, and reads in parallel with the other
                    // rows. Clippy calls it `erasing_op`; collapsing it would hide
                    // which neuron this is.
                    #[allow(clippy::erasing_op)]
                    let vizinho = g.l1[0 * SENSORY_WAVE_SIZE + (ch + 1)];
                    assert_ne!(
                        vizinho,
                        0.0,
                        "seed {seed}: neighboring channel ch{} should not have been zeroed",
                        ch + 1
                    );
                }
            }
        }
    }

    /// P1.8, finding "DEAD_L1_INPUTS zeroes only at birth, evolve()'s mutation does not
    /// reapply the mask": with `p_mutate=1.0` (every weight mutates every generation), the
    /// dead channel's column had a 100% chance of leaving 0.0 as early as generation 1
    /// WITHOUT the fix — this test fails without `filho.zero_dead_inputs()` after the
    /// L1 mutation (`evolve()`, gpu/evolution.rs) and passes with it.
    /// Bug pos-auditoria-s{77,31} (2026-08-14): the mutation seed was
    /// pure f(generation) — two processes with different `--seed`, resuming the
    /// SAME champion, produced byte-identical genomes in every generation (equal md5sum
    /// in gens 0/5/10/15/18). This test fails without the `pop_seed` term in the
    /// seed and passes with it. It also proves the compat: pop_seed=0 is deterministic.
    #[test]
    fn evolve_pop_seeds_diferentes_divergem_e_zero_reproduz() {
        let n = 10;
        let genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        let params = EvoParams {
            p_mutate: 1.0,
            sigma: 0.05,
            n_elite: 1,
            tournament_k: 3,
            congela_l2_cegas: false,
            l2_linhas_lidas: L2_LINHAS_LIDAS,
            mascara_regioes: false,
        };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let a = evolve(&genomes, &fitnesses, &params, 1, 77);
        let b = evolve(&genomes, &fitnesses, &params, 1, 31);
        let a2 = evolve(&genomes, &fitnesses, &params, 1, 77);

        // The elite (index 0) passes cloned in both — compare a MUTATED child.
        assert_ne!(
            a[1].l1, b[1].l1,
            "seeds 77 and 31 produced the same child: pop_seed did not enter the mutation"
        );
        assert_eq!(a[1].l1, a2[1].l1, "same seed stopped being deterministic");
        assert_eq!(
            a[0].l1, b[0].l1,
            "elite cannot depend on the seed (passes without mutation)"
        );
    }

    /// `--l2-congelar-cegas`: L2 rows 5..48 do not mutate; 0..5 mutate normally.
    #[test]
    fn evolve_congela_linhas_cegas_de_l2() {
        let n = 10;
        let genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        let params = EvoParams {
            p_mutate: 1.0,
            sigma: 0.05,
            n_elite: 1,
            tournament_k: 3,
            congela_l2_cegas: true,
            l2_linhas_lidas: L2_LINHAS_LIDAS,
            mascara_regioes: false,
        };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let nova = evolve(&genomes, &fitnesses, &params, 1, 0);
        // Mutated child (index 1): find the parent by comparing the blind rows — they have
        // to be identical to those of SOME original genome (they never mutated).
        let corte = L2_LINHAS_LIDAS * L1_NEURONS;
        let filho = &nova[1];
        let pai = genomes
            .iter()
            .find(|g| g.l2[corte..] == filho.l2[corte..])
            .expect("child's blind rows match no parent: they mutated despite the freeze");
        // And the read rows MUST have mutated (p_mutate=1.0 → 480 weights, all perturbed).
        assert_ne!(
            pai.l2[..corte],
            filho.l2[..corte],
            "read rows did not mutate — the freeze leaked into the wrong range"
        );
    }

    #[test]
    fn evolve_preserva_zero_dos_canais_mortos() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        // Empty list since 2026-08-15 (ch70 refuted by measurement — the channel was
        // ALIVE; thermals implemented in physics_power_c.wgsl). The test becomes a no-op
        // until a genuinely dead channel exists; the machinery stays in the forward path.
        if DEAD_L1_INPUTS.is_empty() {
            return;
        }

        let n = 10;
        let mut genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        // p_mutate=1.0: guarantees that EVERY L1 weight (including those of dead channels,
        // if the mask were not reapplied) undergoes a mutation attempt every generation.
        let params = EvoParams {
            p_mutate: 1.0,
            sigma: 0.05,
            n_elite: 1,
            tournament_k: 3,
            congela_l2_cegas: false,
            l2_linhas_lidas: L2_LINHAS_LIDAS,
            mascara_regioes: false,
        };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        for geracao in 0..10 {
            genomes = evolve(&genomes, &fitnesses, &params, geracao, 0);
            for (gi, g) in genomes.iter().enumerate() {
                for &(ch, _motivo) in DEAD_L1_INPUTS {
                    for neuron in 0..L1_NEURONS {
                        let w = g.l1[neuron * SENSORY_WAVE_SIZE + ch];
                        assert_eq!(
                            w, 0.0,
                            "generation {geracao}, genome {gi}: l1[neuron={neuron}, ch={ch}] = {w}, \
                             expected 0.0 — dead channel resurrected by mutation"
                        );
                    }
                }
            }
        }
    }

    // P0.5, requirement 4 (part 2, "inheritance untouched"): the test that loads a real
    // legacy genome from disk and confirms `load_genome` does not rescale lives in
    // `src/bin/gpu_evolve/tests.rs::genoma_legado_carregado_do_disco_nao_e_reescalado`
    // — it needs `load_genome`/`save_genome`, which live in the `gpu_evolve` binary, not
    // in this crate. The structural guarantee is easy to check by reading: `load_genome`
    // builds `Genome { .. }` via direct struct literal (genome_io.rs, fields listed
    // in the struct's own doc), never calling `Genome::random` nor `spectral_rescale_factor`.

    // ─────────────────────────────────────────────────────────────────────────
    // Tests — SENSORS PHASE (step 1).
    // ─────────────────────────────────────────────────────────────────────────

    /// `SENSOR_DEFAULTS` has to fit inside `SENSOR_RANGES` itself — otherwise the
    /// birth value (used with evolution off, and as the legacy-genome
    /// fallback) would already be born outside the range that mutation/clamp respect when on.
    /// Consistency check between the two tables, not runtime behavior.
    #[test]
    fn sensor_defaults_estao_dentro_das_proprias_faixas() {
        for i in 0..N_SENSOR_GENES {
            let (lo, hi) = SENSOR_RANGES[i];
            let v = SENSOR_DEFAULTS[i];
            assert!(
                v >= lo && v <= hi,
                "gene {i}: default {v} outside the range [{lo}, {hi}]"
            );
        }
    }

    /// `SENSORS_EVOLVE_ON` off (test process default — we do not touch the
    /// atomic here, same care as `genoma_novo_nasce_com_ganho_espectral_perto_de_1`
    /// about not toggling global atomics inside `#[test]`): `Genome::random` produces
    /// `sensors` byte-identical to `SENSOR_DEFAULTS`, under any seed — it is the behavior
    /// that protects the battery in progress (rule from the project's CLAUDE.md).
    #[test]
    fn genoma_novo_nasce_com_sensores_no_padrao_quando_evolucao_desligada() {
        assert!(
            !SENSORS_EVOLVE_ON.load(Ordering::Relaxed),
            "test precondition: SENSORS_EVOLVE_ON should be at the default (off) in this process"
        );
        for seed in [1u64, 2, 3, 42, 999_999] {
            let g = Genome::random(seed);
            assert_eq!(
                g.sensors,
                SENSOR_DEFAULTS.to_vec(),
                "seed {seed}: sensors should be SENSOR_DEFAULTS"
            );
        }
    }

    /// `random_sensors_within_ranges` is a pure function — testable without touching the atomic.
    /// Every drawn gene, under every seed, falls within its own physical range.
    #[test]
    fn sensores_sorteados_ficam_dentro_das_faixas_fisicas() {
        for seed in [1u64, 2, 3, 42, 999_999, 0xDEAD_BEEF] {
            let sensores = random_sensors_within_ranges(seed);
            assert_eq!(sensores.len(), N_SENSOR_GENES);
            for i in 0..N_SENSOR_GENES {
                let (lo, hi) = SENSOR_RANGES[i];
                let v = sensores[i];
                assert!(
                    v >= lo && v <= hi,
                    "seed {seed}, gene {i}: drawn {v} outside the range [{lo}, {hi}]"
                );
            }
        }
    }

    /// `mutate_sensors` is a pure function — testable without touching `SENSORS_EVOLVE_ON`.
    /// Stresses with `p_mutate = 1.0` (mutates every gene every call) for hundreds of
    /// simulated "generations": even accumulating unbounded gaussian noise, the clamp to
    /// `SENSOR_RANGES` guarantees no gene escapes — it is requirement 4 of the task
    /// ("mutation does not leave the ranges").
    #[test]
    fn mutacao_de_sensores_nunca_sai_das_faixas() {
        for seed in [1u64, 2, 3, 42, 999_999] {
            let mut sensores = SENSOR_DEFAULTS.to_vec();
            let mut rng = crate::rng::Lcg::new(seed);
            for _geracao in 0..500 {
                mutate_sensors(&mut sensores, 1.0, &mut rng);
            }
            for i in 0..N_SENSOR_GENES {
                let (lo, hi) = SENSOR_RANGES[i];
                let v = sensores[i];
                assert!(
                    v >= lo && v <= hi,
                    "seed {seed}, gene {i}: {v} outside the range [{lo}, {hi}] after 500 mutations"
                );
            }
        }
    }

    /// With `p_mutate = 0.0`, `mutate_sensors` touches NOTHING — sanity check against a
    /// sign bug in the `rng.chance() < p_mutate` comparison (the same kind of sign bug
    /// already seen in this project in the `slow_probe` LCG, see `src/rng.rs`).
    #[test]
    fn mutacao_com_p_mutate_zero_nao_muda_sensores() {
        let original = SENSOR_DEFAULTS.to_vec();
        let mut sensores = original.clone();
        let mut rng = crate::rng::Lcg::new(123);
        for _ in 0..50 {
            mutate_sensors(&mut sensores, 0.0, &mut rng);
        }
        assert_eq!(sensores, original);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests — P1.7 (asynchronous brain: memory, period/threshold, activation).
    // ─────────────────────────────────────────────────────────────────────────

    /// Central requirement of the task ("each flag at the default reproduces the current
    /// output EXACTLY"): a newborn genome has the six P1.7 fields at their NEUTRAL
    /// value — the value that makes `brain.wgsl` reproduce the pre-change formula
    /// bit for bit (alfa=0 ⇒ h[n]=x[n] with gain 1; period=1 ⇒ recomputes
    /// every tick; threshold=0 ⇒ never zeroes, since `abs(v) < 0.0` is never true;
    /// activation=0 ⇒ tanh). Referenced in the comment of
    /// `BrainParams::mech_off_mask` in `brain.wgsl`.
    #[test]
    fn genoma_novo_tem_mecanismos_p17_neutros() {
        for seed in [1u64, 2, 3, 42, 999_999] {
            let g = Genome::random(seed);
            assert_eq!(
                g.neuron_alpha.len(),
                N_L1_NEURONS + N_L2_NEURONS,
                "seed {seed}"
            );
            assert!(
                g.neuron_alpha.iter().all(|&a| a == 0.0),
                "seed {seed}: alpha should be born 0.0"
            );
            assert_eq!(g.neuron_period_l1.len(), N_L1_NEURONS, "seed {seed}");
            assert!(
                g.neuron_period_l1.iter().all(|&p| p == 1.0),
                "seed {seed}: period should be born 1.0"
            );
            assert_eq!(g.neuron_threshold_l1.len(), N_L1_NEURONS, "seed {seed}");
            assert!(
                g.neuron_threshold_l1.iter().all(|&t| t == 0.0),
                "seed {seed}: threshold should be born 0.0"
            );
            assert_eq!(
                g.neuron_activation.len(),
                N_L1_NEURONS + N_L2_NEURONS,
                "seed {seed}"
            );
            assert!(
                g.neuron_activation.iter().all(|&a| a == 0.0),
                "seed {seed}: activation should be born 0.0 (tanh)"
            );
        }
    }

    /// The amplitude-gain math of mechanism 1 (`neuron_mem_gain` in
    /// `brain.wgsl`), mirrored here in pure Rust so it can be tested without a GPU — the
    /// same pattern as `out_stage`/`population_std_dev` in `math/activations.rs`.
    /// `gain(0) == 1` is what guarantees that alfa=0 changes NOTHING in the amplitude.
    fn neuron_mem_gain(alfa: f32) -> f32 {
        ((1.0 + alfa) / (1.0 - alfa).max(1e-3)).sqrt()
    }

    #[test]
    fn neuron_mem_gain_em_alfa_zero_e_identidade() {
        assert_eq!(neuron_mem_gain(0.0), 1.0);
    }

    /// The gain grows with alpha (compensates the amplitude attenuation — GAP 11 of the
    /// project's CLAUDE.md) and stays FINITE at `evolve()`'s clamp ceiling (0.95).
    #[test]
    fn neuron_mem_gain_cresce_e_fica_finito_no_teto_do_clamp() {
        let g_baixo = neuron_mem_gain(0.2);
        let g_alto = neuron_mem_gain(0.95);
        assert!(
            g_alto > g_baixo,
            "gain should grow with alpha: {g_baixo} vs {g_alto}"
        );
        assert!(
            g_alto.is_finite(),
            "gain at the clamp ceiling (0.95) should be finite, got {g_alto}"
        );
        // sqrt((1+0.95)/(1-0.95)) = sqrt(1.95/0.05) = sqrt(39) ≈ 6.245
        assert!(
            (g_alto - 39.0f32.sqrt()).abs() < 1e-4,
            "gain at alfa=0.95: {g_alto}, expected ≈6.245"
        );
    }

    /// `evolve()`, with the four P1.7 atomics at their default (`true` — mutation
    /// ON), mutates the six fields across several generations but NEVER leaves the
    /// physically valid ranges: alfa∈[0,0.95], period∈[1,8] (integer),
    /// threshold∈[0,0.9], activation∈{0,1,2,3} — AND, in the SAME test (not in two
    /// separate ones), the control arm: with the four atomics OFF,
    /// `evolve()` NEVER touches the six P1.7 fields, they stay bit-identical to the
    /// birth neutral for however many generations it runs.
    ///
    /// MERGED into a single `#[test]` on purpose: the four atomics
    /// (`NEURON_*_MUTATE_ON`) are process globals, and `cargo test` runs
    /// tests of the same binary in THREADS of the same process — two separate `#[test]`s,
    /// one reading "should be true" and another toggling to `false`
    /// and back, run a real risk of crossing (measured: the first version
    /// of this test, split, failed intermittently for that reason — R8 of the
    /// project's CLAUDE.md). One single test, in one single thread, eliminates the race:
    /// first it measures the default (mutating), then turns on the control arm
    /// (freezes), with no other thread able to observe an intermediate
    /// state of these atomics.
    #[test]
    fn evolve_muta_mecanismos_p17_e_flags_sem_x_congelam_no_neutro() {
        assert!(
            NEURON_MEM_MUTATE_ON.load(Ordering::Relaxed),
            "precondition: default should be true"
        );
        assert!(
            NEURON_PERIOD_MUTATE_ON.load(Ordering::Relaxed),
            "precondition: default should be true"
        );
        assert!(
            NEURON_THRESHOLD_MUTATE_ON.load(Ordering::Relaxed),
            "precondition: default should be true"
        );
        assert!(
            NEURON_ACTIVATION_MUTATE_ON.load(Ordering::Relaxed),
            "precondition: default should be true"
        );

        // ── Part 1: default (mutation ON) — ranges respected + something changed ──
        let n = 12;
        let mut genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        let params = EvoParams {
            p_mutate: 1.0,
            sigma: 0.05,
            n_elite: 1,
            tournament_k: 3,
            congela_l2_cegas: false,
            l2_linhas_lidas: L2_LINHAS_LIDAS,
            mascara_regioes: false,
        };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        for geracao in 0..40 {
            genomes = evolve(&genomes, &fitnesses, &params, geracao, 0);
            for g in &genomes {
                for &a in &g.neuron_alpha {
                    assert!(
                        (0.0..=0.95).contains(&a),
                        "generation {geracao}: alpha {a} outside [0, 0.95]"
                    );
                }
                for &p in &g.neuron_period_l1 {
                    assert!(
                        (1.0..=8.0).contains(&p),
                        "generation {geracao}: period {p} outside [1, 8]"
                    );
                    assert_eq!(
                        p,
                        p.round(),
                        "generation {geracao}: period {p} should be an integer"
                    );
                }
                for &t in &g.neuron_threshold_l1 {
                    assert!(
                        (0.0..=0.9).contains(&t),
                        "generation {geracao}: threshold {t} outside [0, 0.9]"
                    );
                }
                for &act in &g.neuron_activation {
                    assert!(
                        (0.0..=3.0).contains(&act) && act == act.floor(),
                        "generation {geracao}: activation id {act} should be an integer in [0,3]"
                    );
                }
            }
        }

        // After 40 generations with p_mutate=1.0 (every mutation always fires), at
        // least SOME value should have left neutral — otherwise the test above
        // would be vacuously true (everything stays 0.0/1.0/0.0/0.0 and "never leaves
        // the range" would be trivial, not a proof that mutation works).
        let algum_alfa_mudou = genomes
            .iter()
            .any(|g| g.neuron_alpha.iter().any(|&a| a != 0.0));
        let algum_periodo_mudou = genomes
            .iter()
            .any(|g| g.neuron_period_l1.iter().any(|&p| p != 1.0));
        let algum_limiar_mudou = genomes
            .iter()
            .any(|g| g.neuron_threshold_l1.iter().any(|&t| t != 0.0));
        let alguma_ativacao_mudou = genomes
            .iter()
            .any(|g| g.neuron_activation.iter().any(|&a| a != 0.0));
        assert!(
            algum_alfa_mudou,
            "40 generations with p_mutate=1.0 and no alpha changed — mutation is not running"
        );
        assert!(
            algum_periodo_mudou,
            "40 generations with p_mutate=1.0 and no period changed — mutation is not running"
        );
        assert!(
            algum_limiar_mudou,
            "40 generations with p_mutate=1.0 and no threshold changed — mutation is not running"
        );
        assert!(
            alguma_ativacao_mudou,
            "40 generations and no activation changed — discrete mutation is not running"
        );

        // ── Part 2: control arm (`--sem-*`) — freezes at neutral ──────────
        // `let _guard` restores the atomics to the default even if the rest of the test
        // fails midway, so it does not leak `false` into tests running AFTER
        // this one in the same process (even though today no other test touches these
        // atomics — a defense against the next test that comes to touch them).
        struct RestauraAtomics;
        impl Drop for RestauraAtomics {
            fn drop(&mut self) {
                NEURON_MEM_MUTATE_ON.store(true, Ordering::Relaxed);
                NEURON_PERIOD_MUTATE_ON.store(true, Ordering::Relaxed);
                NEURON_THRESHOLD_MUTATE_ON.store(true, Ordering::Relaxed);
                NEURON_ACTIVATION_MUTATE_ON.store(true, Ordering::Relaxed);
            }
        }
        let _guard = RestauraAtomics;
        NEURON_MEM_MUTATE_ON.store(false, Ordering::Relaxed);
        NEURON_PERIOD_MUTATE_ON.store(false, Ordering::Relaxed);
        NEURON_THRESHOLD_MUTATE_ON.store(false, Ordering::Relaxed);
        NEURON_ACTIVATION_MUTATE_ON.store(false, Ordering::Relaxed);

        let n2 = 10;
        let mut genomes2: Vec<Genome> = (0..n2).map(|i| Genome::random(i as u64 + 500)).collect();
        let fitnesses2: Vec<f64> = (0..n2).map(|i| i as f64).collect();

        for geracao in 0..20 {
            genomes2 = evolve(&genomes2, &fitnesses2, &params, geracao, 0);
        }

        for g in &genomes2 {
            assert!(
                g.neuron_alpha.iter().all(|&a| a == 0.0),
                "--sem-memoria-neuronal should freeze alpha at 0.0"
            );
            assert!(
                g.neuron_period_l1.iter().all(|&p| p == 1.0),
                "--sem-periodo-neuronal should freeze period at 1.0"
            );
            assert!(
                g.neuron_threshold_l1.iter().all(|&t| t == 0.0),
                "--sem-limiar-neuronal should freeze threshold at 0.0"
            );
            assert!(
                g.neuron_activation.iter().all(|&a| a == 0.0),
                "--sem-ativacao-heterogenea should freeze activation at 0.0"
            );
        }
    }

    /// Control arm of mechanism 2: with ALL periods at 1.0 (the birth
    /// neutral), the simulator has to return ALL neurons firing
    /// on EVERY tick — ZERO deviation (nothing varies), mean = n_beings×96. Proves that the
    /// function reproduces "today" exactly when the period is at the default.
    #[test]
    fn simulate_l1_fire_counts_com_periodo_neutro_dispara_todo_neuronio_todo_tick() {
        let n_beings = 5;
        let periods = vec![1.0f32; n_beings * N_L1_NEURONS];
        let (media, desvio) = simulate_l1_fire_counts(&periods, n_beings, 100);
        assert_eq!(
            media,
            (n_beings * N_L1_NEURONS) as f64,
            "period=1 should fire 100% always"
        );
        assert_eq!(desvio, 0.0, "no variation possible with uniform period=1");
    }

    /// With uniform period=2, the MEAN over time is 50% of the population —
    /// but the deviation is NOT zero: since all neurons share the SAME phase
    /// (shared counter), they fire in LOCKSTEP — even tick = ALL
    /// fire, odd tick = NONE fire. It is a step (0 or N), not a constant
    /// value — real asynchrony only appears when the PERIODS
    /// differ between neurons (see the mixed-population test, below), not
    /// when everyone shares the same uniform period.
    #[test]
    fn simulate_l1_fire_counts_com_periodo_dois_alterna_entre_zero_e_total() {
        let n_beings = 4;
        let total = (n_beings * N_L1_NEURONS) as f64; // 384
        let periods = vec![2.0f32; n_beings * N_L1_NEURONS];
        let (media, desvio) = simulate_l1_fire_counts(&periods, n_beings, 200);
        assert_eq!(media, total / 2.0, "period=2 should fire 50% on average");
        // Two-point distribution (0 or total, 50/50): deviation = total/2 exactly.
        assert_eq!(
            desvio,
            total / 2.0,
            "0/total step should have deviation = total/2"
        );
    }

    /// MIXED population (half period=1, half period=4): the mean falls between
    /// the two extremes and the deviation becomes NON-ZERO (the per-tick total varies
    /// as the multiples of 4 align or not) — sanity check that the
    /// function truly sums heterogeneous contributions, not just replicates one case.
    #[test]
    fn simulate_l1_fire_counts_populacao_mista_tem_desvio_no_zero() {
        let n_beings = 2;
        let mut periods = vec![1.0f32; N_L1_NEURONS]; // being 0: all period 1
        periods.extend(vec![4.0f32; N_L1_NEURONS]); // being 1: all period 4
        let (media, desvio) = simulate_l1_fire_counts(&periods, n_beings, 400);
        // Being 0 contributes 96 always; being 1 contributes 96 every 4 ticks, 0 in the others —
        // expected mean = 96 + 96*0.25 = 120.
        assert!(
            (media - 120.0).abs() < 1.0,
            "expected mean ≈120, got {media}"
        );
        assert!(
            desvio > 0.0,
            "mixed population should have deviation > 0, got {desvio}"
        );
    }

    /// `pack_neuron_meta` produces the EXACT layout `brain.wgsl` expects: per
    /// genome, 480 floats in the six blocks in the order l1_alpha·l1_period·
    /// l1_threshold·l1_activation·l2_alpha·l2_activation — not interleaved per
    /// being (unlike `pack_taus`/`pack_sensors`, here L1 and L2 come from TWO
    /// genome vectors that need to be stitched in the right order).
    #[test]
    fn pack_neuron_meta_produz_layout_esperado_pelo_shader() {
        let mut g = Genome::random(7);
        // Marks each block with a unique, identifiable value, away from neutral.
        for (i, a) in g.neuron_alpha.iter_mut().enumerate() {
            *a = 1000.0 + i as f32;
        }
        for (i, p) in g.neuron_period_l1.iter_mut().enumerate() {
            *p = 2000.0 + i as f32;
        }
        for (i, t) in g.neuron_threshold_l1.iter_mut().enumerate() {
            *t = 3000.0 + i as f32;
        }
        for (i, act) in g.neuron_activation.iter_mut().enumerate() {
            *act = 4000.0 + i as f32;
        }

        let packed = Genome::pack_neuron_meta(&[g.clone()]);
        assert_eq!(packed.len(), N_L1_NEURONS * 4 + N_L2_NEURONS * 2);

        // l1_alpha = neuron_alpha[0..96] = 1000..1096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[i], 1000.0 + i as f32, "l1_alpha[{i}]");
        }
        // l1_period = neuron_period_l1[0..96] = 2000..2096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[96 + i], 2000.0 + i as f32, "l1_period[{i}]");
        }
        // l1_threshold = neuron_threshold_l1[0..96] = 3000..3096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[192 + i], 3000.0 + i as f32, "l1_threshold[{i}]");
        }
        // l1_activation = neuron_activation[0..96] = 4000..4096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[288 + i], 4000.0 + i as f32, "l1_activation[{i}]");
        }
        // l2_alpha = neuron_alpha[96..144] = 1096..1144
        for i in 0..N_L2_NEURONS {
            assert_eq!(
                packed[384 + i],
                1000.0 + (N_L1_NEURONS + i) as f32,
                "l2_alpha[{i}]"
            );
        }
        // l2_activation = neuron_activation[96..144] = 4096..4144
        for i in 0..N_L2_NEURONS {
            assert_eq!(
                packed[432 + i],
                4000.0 + (N_L1_NEURONS + i) as f32,
                "l2_activation[{i}]"
            );
        }
    }
}
