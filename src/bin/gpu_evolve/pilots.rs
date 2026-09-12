//! Diagnostic pilots: purely geometric/sensory controllers that REPLACE the brain
//! behind a flag, to separate blame on the BODY from blame on the BRAIN.
//!
//! - **Sensory pilot** (`--pilot-sensor`): drives only by wall rays — measures the
//!   ceiling of the available sensory information.
//! - **Ghost pilot** (`--pilot`): drives along the racing line (an oracle the being
//!   does not have) — measures whether the BODY can complete the track, isolating the
//!   brain's competence.
//!
//! Neither function here is used in production (normal fitness runs with gain
//! 0.0, see `statics::PILOTO_GANHO`/`statics::PILOTO_SENSOR`) — they are diagnostic
//! instruments, called from inside `context.rs::GpuCtx::tick()`.
//!
//! `voxel_at` arrives as a closure (`impl Fn`) instead of accessing `GpuCtx` directly
//! because these functions are pure — testable without a GPU — and because
//! `GpuCtx::voxel_at` needs `&self.grid_cpu` (it cannot become a free `fn` without the
//! calling being's context, see note R of REFACTOR_PLAN.md about the triplicated
//! `voxel_at` closures).

/// Ray to the wall in a direction `(dx, dz)` starting from `(px, pz)`, in metres
/// (1 m step, ceiling `cap`). Used by both pilots.
#[inline]
fn raio(px: f32, pz: f32, dx: f32, dz: f32, cap: f32, voxel_at: &impl Fn(f32, f32) -> u32) -> f32 {
    let mut d = cap;
    let mut t = 1.0_f32;
    while t <= cap {
        let v = voxel_at(px + dx * t, pz + dz * t);
        if v == 2 || v == 3 {
            d = t;
            break;
        }
        t += 1.0;
    }
    d
}

/// Sensory pilot command: a "twenty-line" controller that drives only by wall rays
/// (laterals + ±45° diagonals), trying to centre itself and anticipate curves.
/// Returns `(steer, throttle, brake)`.
///
/// # Why
///
/// Leonardo: *"run a battery where there is no brain, the car must survive by
/// sensors, see the distance of the walls on both sides, always trying to centre
/// itself in the middle of the track and take the curves correctly."* The earlier
/// ghost pilot used the racing line — an oracle the being does NOT have, so it
/// measured the body, not the information. This one uses only wall rays, exactly what
/// the being's retina sees.
pub fn piloto_sensorial(
    ganho: f32,
    px: f32,
    pz: f32,
    fx: f32,
    fz: f32,
    perp_x: f32,
    perp_z: f32,
    speed: f32,
    fwd_dist: f32,
    voxel_at: &impl Fn(f32, f32) -> u32,
) -> (f32, f32, f32) {
    // Pure laterals: centring.
    let esq = raio(px, pz, perp_x, perp_z, 25.0, voxel_at);
    let dir = raio(px, pz, -perp_x, -perp_z, 25.0, voxel_at);
    // ±45° diagonals: anticipation. The curve shows up in them BEFORE it shows up
    // in the laterals — that is what allows turning before scraping the wall.
    let k = std::f32::consts::FRAC_1_SQRT_2;
    let diag_e = raio(
        px,
        pz,
        fx * k + perp_x * k,
        fz * k + perp_z * k,
        25.0,
        voxel_at,
    );
    let diag_d = raio(
        px,
        pz,
        fx * k - perp_x * k,
        fz * k - perp_z * k,
        25.0,
        voxel_at,
    );

    // Centring error, normalised: +1 = glued to one side.
    let centro = (dir - esq) / (dir + esq).max(1.0);
    // Anticipation error: which way the track opens up ahead.
    let abertura = (diag_d - diag_e) / (diag_d + diag_e).max(1.0);

    // Anticipation weighs more than centring: correcting only by the lateral is
    // reacting after already being wrong, and on a 28 m track that arrives late in the
    // 51 m radius curve.
    let steer = (ganho * (centro * 0.4 + abertura * 1.0)).clamp(-1.0, 1.0);

    // Speed from the critical space: uses the MINIMUM between the frontal distance and
    // the narrow diagonals. The diagonals detect the curve's side wall BEFORE the
    // frontal ray — that is what allows braking before crashing.
    let critico = fwd_dist.min(diag_e).min(diag_d);
    let assimetria = (diag_e - diag_d).abs() / (diag_e + diag_d).max(1.0);
    // critico falls when there is a wall nearby; assimetria rises when there is a curve.
    let sinal_curva = (1.0 - assimetria).clamp(0.0, 1.0);
    let livre = (critico / 25.0).clamp(0.0, 1.0) * sinal_curva;
    let alvo = 5.0 + livre * 13.0; // 5 m/s in the tight curve, 18 on the straight
    let thr = if speed < alvo { 0.85 } else { 0.0 };
    let brk = if speed > alvo * 1.05 { 0.7 } else { 0.0 };
    (steer, thr, brk)
}

/// Minimum fan-ray distance (7 directions: forward + ±15°/30°/45°) around the being.
/// Feeds the ghost pilot's anticipatory brake. The curve shows up as a diagonal wall
/// BEFORE it shows up frontally — the fan detects the curve before the pure ray does.
pub fn fan_ray_d_min(
    px: f32,
    pz: f32,
    fx: f32,
    fz: f32,
    voxel_at: &impl Fn(f32, f32) -> u32,
) -> f32 {
    let (pxp, pzp) = (-fz, fx); // left perpendicular
    let k15 = 0.9659_f32;
    let s15 = 0.2588_f32;
    let k30 = 0.8660_f32;
    let s30 = 0.5000_f32;
    // cos(45°) = sin(45°) = 1/sqrt(2); use the constant instead of a 4-digit literal.
    let k45 = std::f32::consts::FRAC_1_SQRT_2;
    let s45 = std::f32::consts::FRAC_1_SQRT_2;
    let d0 = raio(px, pz, fx, fz, 30.0, voxel_at);
    let dl15 = raio(
        px,
        pz,
        fx * k15 + pxp * s15,
        fz * k15 + pzp * s15,
        30.0,
        voxel_at,
    );
    let dr15 = raio(
        px,
        pz,
        fx * k15 - pxp * s15,
        fz * k15 - pzp * s15,
        30.0,
        voxel_at,
    );
    let dl30 = raio(
        px,
        pz,
        fx * k30 + pxp * s30,
        fz * k30 + pzp * s30,
        30.0,
        voxel_at,
    );
    let dr30 = raio(
        px,
        pz,
        fx * k30 - pxp * s30,
        fz * k30 - pzp * s30,
        30.0,
        voxel_at,
    );
    let dl45 = raio(
        px,
        pz,
        fx * k45 + pxp * s45,
        fz * k45 + pzp * s45,
        25.0,
        voxel_at,
    );
    let dr45 = raio(
        px,
        pz,
        fx * k45 - pxp * s45,
        fz * k45 - pzp * s45,
        25.0,
        voxel_at,
    );
    d0.min(dl15)
        .min(dr15)
        .min(dl30)
        .min(dr30)
        .min(dl45)
        .min(dr45)
}

/// Ghost pilot pure-pursuit command: aims at a point `L` metres ahead on the racing
/// line and corrects by heading (not by lateral deviation — this avoids the
/// `lookahead_offset` bug that mixes longitudinal with lateral error when the car is
/// behind the lookahead point). Adaptive `L`: more distance at high speed to smooth it
/// out. `d_min` = output of `fan_ray_d_min`, used for the anticipatory curve brake.
///
/// Returns `(steer, throttle, brake)`.
#[allow(clippy::too_many_arguments)]
pub fn piloto_fantasma_comando(
    ganho: f32,
    thr_base: f32,
    pos_x: f32,
    pos_z: f32,
    fwd_x: f32,
    fwd_z: f32,
    tgt_x: f32,
    tgt_z: f32,
    speed_now: f32,
    d_min: f32,
) -> (f32, f32, f32) {
    let dx = tgt_x - pos_x;
    let dz = tgt_z - pos_z;
    let dist_tgt = (dx * dx + dz * dz).sqrt().max(0.1);
    // Cross product (fwd × dir_to_target): positive = target to the left.
    let cross = fwd_x * (dz / dist_tgt) - fwd_z * (dx / dist_tgt);
    // Target to the left (cross > 0) → steer left (negative in the game).
    // Gain proportional to the angle; the ±0.7 cap avoids skidding.
    let steer = (ganho * (-cross * 1.2)).clamp(-0.7, 0.7);

    // Physical v_max for R=16m, mu=1.6: 15.9 m/s. Target 14 m/s with margin.
    let livre_geo = (d_min / 30.0).clamp(0.0, 1.0);
    let v_alvo = 6.0 + livre_geo * 10.0; // 6 m/s near a wall, 16 m/s free
    let (thr, brk) = if speed_now > v_alvo + 0.5 {
        (0.0, ((speed_now - v_alvo) / 8.0).clamp(0.15, 0.7))
    } else if speed_now < v_alvo - 1.5 {
        (thr_base, 0.0)
    } else {
        (thr_base * 0.4, 0.0)
    };
    (steer, thr, brk)
}
