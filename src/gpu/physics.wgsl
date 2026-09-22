// VEHICLE PHYSICS ON THE GPU — one thread per car, single kernel per simulation step.
//
// # Design decision: single kernel vs four separate kernels
//
// The CPU runs suspension → powertrain → collision → integration in sequence, and EACH PHASE
// reads the previous phase's result FOR THE SAME CAR. That rules out asynchronous pipelines
// because the dependency is local (car i depends only on itself). With a single kernel:
//
//   1. No global memory barriers between phases — each thread loads its car's state
//      into registers, runs the 4 phases in sequence and writes back a single time.
//   2. No overhead of 4 dispatches + 4 bind groups + 4 syncs.
//   3. The GPU has ~1800 f32 registers per RDNA2 workgroup; one car uses ~150 floats.
//      Not measured — conservative estimate. Register pressure should not be a bottleneck.
//
// With 60 beings it fits in 1 workgroup of 64 threads (threads 60-63 exit at the initial guard).
// The compute saturation point only appears above ~512 threads; at 60 the bottleneck is
// host↔GPU round-trip latency, not ALU. The main gain comes from resident state.
//
// # Arithmetic divergence risks vs CPU
//
// 1. WGSL `atan` and `sin`: IEEE-754 single-precision, faithful to ±1 ULP in most
//    Vulkan/GL implementations. The GPU may use a native instruction that differs from x86 libm.
//    Sensitive spot: Pacejka (sin(c * atan(b * sr))) and slip_angle (atan2). Documented
//    in-situ.
//
// 2. `sqrt` / `length`: ±1 ULP vs the CPU's sqrtf. Critical in the Kamm circle.
//
// 3. Force accumulation order: same as CPU (accumulates in local variables, applies at the end).
//
// 4. Bilinear heightmap interpolation: floor() GPU vs CPU — both follow IEEE 60559. OK.
//
// 5. DEAD ASSUMPTION at physics.rs line 631:
//    `yaw_inertia = mass * (wheel_base/2)^2`
//    It is the moment of inertia of a point mass; ignores track_width and the real distribution.
//    The comment "correct moment of inertia" is optimistic. We replicate the same formula.
//
// 6. POSSIBLE BUG in the CPU physics.rs line 467:
//    `split_left = 0.5 + lock_factor * signum(omega_R - omega_L)`
//    split_left is not clamped to [0, 1] — can be >1.0 with a high lock_factor.
//    split_right = 1.0 - split_left can be negative. We replicate faithfully.
//
// 7. AERO DRAG DUPLICATION on the CPU: powertrain_and_tire_system applies drag (phase E) AND
//    integration_system reapplies it (line 717-721). We replicate both.
//
// 8. COLLISION YAW KICK: the CPU (line 687) adds directly to body.angular_velocity.y
//    — not via torque. Here we use accumulated av_y_col, added BEFORE integration.

// ============================================================
// === INPUT/OUTPUT STRUCTS                                 ===
// ============================================================

// Full state of one car for reading. Mirror of EstadoFisicoGpu in fisica.rs.
// All vec3 carry f32 padding for the 16-byte alignment WGSL requires.
struct CarInput {
    // --- PhysicsBody ---
    vel:        vec3<f32>,
    av_y:       f32,         // angular_velocity.y (rad/s)
    // 4 wheels × 2 vec4 = 8 vec4. Wheel i: wheels[i*2] = [omega, slip_r, slip_a, normal_load]
    //                                      wheels[i*2+1] = [tire_temp, susp_comp, prev_comp, pad]
    wheels:     array<vec4<f32>, 8>,
    // --- Transform ---
    pos:        vec3<f32>,
    _pad_pos:   f32,
    fwd:        vec3<f32>,   // transform.right() in Bevy = "forward" in physics.rs
    _pad_fwd:   f32,
    lat:        vec3<f32>,   // transform.back() in Bevy = "lateral" in physics.rs
    _pad_lat:   f32,
    // --- EngineComponent ---
    max_torque:      f32,
    current_rpm:     f32,
    max_rpm:         f32,
    throttle:        f32,
    brake_cmd:       f32,
    steer:           f32,
    drive_torque_out:f32,
    current_gear:    u32,   // 0=reverse, 1=neutral, 2=1st … 6=6th
    shift_timer:     f32,
    is_stalled:      u32,
    starter_timer:   f32,
    flywheel_inertia:f32,
    engine_temp:     f32,
    oil_temp:        f32,
    // --- VehicleChassis ---
    mass:            f32,
    drag_coeff:      f32,
    wheel_base:      f32,
    cg_height:       f32,
    track_width:     f32,
    lsd_lock_ratio:  f32,
    brake_torque_max:f32,
    drivetrain:      u32,   // 0=FWD, 1=RWD, 2=AWD
    // --- TireComponent ---
    mu_peak:             f32,
    cornering_stiffness: f32,
    wheel_radius:        f32,
    wheel_inertia:       f32,
    optimal_temp:        f32,
    // --- SuspensionComponent ---
    susp_stiffness:  f32,
    susp_damping:    f32,
    susp_rest:       f32,
    susp_max_travel: f32,
    // PADDING: offset after susp_max_travel = 316 (not a multiple of 16).
    // wind: vec3<f32> requires alignment 16 → insert 4 bytes here to reach offset 320.
    _pad_before_wind: f32,
    // --- Environment (tick snapshot) ---
    wind:            vec3<f32>,
    _pad_wind:       f32,
    terrain_grip:    f32,
    env_grip:        f32,
    aquaplaning:     f32,
    // --- Config + dt ---
    config_flags:    u32,
    dt:              f32,
    _pad_dt:         f32,
}

struct CarOutput {
    vel:             vec3<f32>,
    av_y:            f32,
    wheels:          array<vec4<f32>, 8>,
    pos:             vec3<f32>,
    _pad_pos:        f32,
    fwd:             vec3<f32>,
    _pad_fwd:        f32,
    lat:             vec3<f32>,
    _pad_lat:        f32,
    current_rpm:     f32,
    drive_torque_out:f32,
    shift_timer:     f32,
    is_stalled:      u32,
    starter_timer:   f32,
    engine_temp:     f32,
    oil_temp:        f32,
    _pad_eng:        f32,
}

struct WorldParams {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    _pad:      f32,
}

@group(0) @binding(0) var<uniform>             params:  WorldParams;
@group(0) @binding(1) var<storage, read>       grid:   array<u32>;
@group(0) @binding(2) var<storage, read>       hmap:    array<f32>;
@group(0) @binding(3) var<storage, read>       input: array<CarInput>;
@group(0) @binding(4) var<storage, read_write> output:   array<CarOutput>;

// ============================================================
// === CONSTANTS                                            ===
// ============================================================

const GRAVITY:      f32 = 9.81;
const RHO_AR:       f32 = 1.225;   // kg/m³ air density at sea level
const FRONTAL_AREA: f32 = 2.2;     // m² GT3 frontal area (not measured, assumed in physics.rs)
const PI2:          f32 = 6.283185307179586;
const BASE_ENGINE_TORQUE_NM: f32 = 620.0;

// ============================================================
// === WORLD HELPERS                                        ===
// ============================================================

// Identical to get_voxel_at in world.rs: round() + index [z*size+x].
// Returns Wall (2) out of bounds — same as CPU.
fn cell_round(v: f32) -> f32 { return sign(v) * floor(abs(v) + 0.5); }
fn voxel_at(wx: f32, wz: f32) -> u32 {
    // Half away from zero like Rust f32::round; WGSL round() is half to even.
    let x = i32(cell_round(wx / params.cell_size));
    let z = i32(cell_round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if (x >= 0 && x < n && z >= 0 && z < n) {
        return grid[u32(z * n + x)];
    }
    return 2u; // Wall
}

// Identical to get_height_at in world.rs: bilinear interpolation.
// POSSIBLE DIVERGENCE: floor() is IEEE 60559 specified in both — OK for normal inputs.
fn height_at(wx: f32, wz: f32) -> f32 {
    let fx = wx / params.cell_size;
    let fz = wz / params.cell_size;
    let x0 = i32(floor(fx));
    let z0 = i32(floor(fz));
    let tx = fx - f32(x0);
    let tz = fz - f32(z0);
    let n  = i32(params.grid_size);

    var h00 = 0.0; var h10 = 0.0; var h01 = 0.0; var h11 = 0.0;
    if (x0 >= 0 && x0 < n && z0 >= 0 && z0 < n)                 { h00 = hmap[u32(z0 * n + x0)]; }
    if ((x0+1) >= 0 && (x0+1) < n && z0 >= 0 && z0 < n)         { h10 = hmap[u32(z0 * n + (x0+1))]; }
    if (x0 >= 0 && x0 < n && (z0+1) >= 0 && (z0+1) < n)         { h01 = hmap[u32((z0+1) * n + x0)]; }
    if ((x0+1) >= 0 && (x0+1) < n && (z0+1) >= 0 && (z0+1) < n) { h11 = hmap[u32((z0+1) * n + (x0+1))]; }

    let top = h00 * (1.0 - tx) + h10 * tx;
    let bot = h01 * (1.0 - tx) + h11 * tx;
    return top * (1.0 - tz) + bot * tz;
}

// normalize_or_zero: same as Bevy Vec3::normalize_or_zero.
// Separate from the builtin `normalize` to avoid NaN when v = 0.
fn nor0(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l < 1e-9) { return vec3<f32>(0.0, 0.0, 0.0); }
    return v / l;
}

// Extracts bit `bit` from config_flags. Takes the value directly (not a pointer to the struct)
// to avoid depending on EntradaCarro's address space.
fn flag(flags: u32, bit: u32) -> bool {
    return (flags >> bit & 1u) != 0u;
}

// Transmission ratio per gear. Index: 0=reverse, 1=neutral, 2=1st..6=6th.
// Values identical to the CPU's `gears` array (physics.rs line 263).
fn gear_ratio(g: u32) -> f32 {
    switch g {
        case 0u: { return -3.5; }
        case 1u: { return  0.0; }
        case 2u: { return  3.8; }
        case 3u: { return  2.6; }
        case 4u: { return  1.9; }
        case 5u: { return  1.45; }
        default: { return  1.15; }
    }
}

const FINAL_DRIVE: f32 = 3.7;

// ============================================================
// === MAIN KERNEL                                          ===
// ============================================================

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let being = gid.x;
    if (being >= params.n_beings) { return; }

    // Local copy of the input (single read of the storage buffer — registers operate afterwards)
    var c = input[being];

    let dt_safe = max(c.dt, 0.001);  // same guard as the CPU: `time.delta_secs().max(0.001)`

    // ============================================================
    // === PHASE 1: RESET FORCES                                ===
    // ============================================================
    var forces = vec3<f32>(0.0, -GRAVITY * c.mass, 0.0);
    var torques_y = 0.0;  // only yaw matters in the 2.5D model

    // ============================================================
    // === PHASE 2: SUSPENSION + NORMAL LOAD                   ===
    // ============================================================
    let speed = length(c.vel);
    let aero_downforce = select(0.0, 3.0 * speed * speed, flag(c.config_flags,3u));
    let downforce_per_wheel = aero_downforce * 0.25;
    let static_load = c.mass * GRAVITY / 4.0;

    let fwd = c.fwd;
    let rgt = -c.lat;   // real right = -back() in Bevy

    let half_wb = c.wheel_base * 0.5;
    let half_tw = c.track_width * 0.5;

    // Wheel offsets [FL, FR, RL, RR] — same as CPU (suspension_and_load_system)
    var wheel_offsets: array<vec3<f32>, 4>;
    wheel_offsets[0] = fwd * half_wb - rgt * half_tw;  // FL
    wheel_offsets[1] = fwd * half_wb + rgt * half_tw;  // FR
    wheel_offsets[2] = -fwd * half_wb - rgt * half_tw; // RL
    wheel_offsets[3] = -fwd * half_wb + rgt * half_tw; // RR

    var normal_loads: array<f32, 4>;
    var susp_comps:   array<f32, 4>;
    var prev_comps:   array<f32, 4>;
    var total_susp_load = 0.0;

    for (var i: u32 = 0u; i < 4u; i++) {
        let wheel_pos = c.pos + wheel_offsets[i];
        let ground_y  = height_at(wheel_pos.x, wheel_pos.z);
        let height_above = wheel_pos.y - ground_y;
        let compression  = c.susp_rest - height_above;
        let prev_c = c.wheels[i * 2u + 1u].z;  // prev_compression

        if (compression > 0.0 && compression < c.susp_max_travel) {
            let spring_f = c.susp_stiffness * compression;
            let comp_vel = (compression - prev_c) / dt_safe;
            let damper_f = c.susp_damping * comp_vel;
            let total_f  = max(spring_f + damper_f, 0.0) + downforce_per_wheel;
            let clamped  = clamp(total_f, static_load * 0.2, static_load * 6.0);

            forces.y      += clamped;
            normal_loads[i] = clamped;
            susp_comps[i]   = compression;
        } else {
            normal_loads[i] = 0.0;
            susp_comps[i]   = 0.0;
        }
        prev_comps[i]       = compression;
        total_susp_load    += normal_loads[i];
    }

    // Fallback: no load on any wheel → static weight per wheel
    let total_weight = c.mass * GRAVITY;
    if (total_susp_load < 100.0) {
        for (var i: u32 = 0u; i < 4u; i++) {
            normal_loads[i] = total_weight * 0.25;
        }
    }

    // ============================================================
    // === PHASE 3: POWERTRAIN + TYRES + AERO                  ===
    // ============================================================
    let forward_dir = fwd;
    let lateral_dir = c.lat;    // Bevy's back() — same as CPU
    let forward_vel = dot(c.vel, forward_dir);
    let lateral_vel = dot(c.vel, lateral_dir);

    let gratio        = gear_ratio(c.current_gear);
    let total_ratio   = gratio * FINAL_DRIVE;
    let total_ratio_abs = abs(total_ratio);

    // Mutable engine state (copied into local vars)
    var is_stalled      = c.is_stalled;
    var starter_timer   = c.starter_timer;
    var shift_timer     = c.shift_timer;
    var current_rpm     = c.current_rpm;
    var engine_temp     = c.engine_temp;
    var oil_temp        = c.oil_temp;

    // Mutable wheel state
    var omegas:      array<f32, 4>;
    var tire_temps:  array<f32, 4>;
    var slip_ratios: array<f32, 4>;
    var slip_angles: array<f32, 4>;
    for (var i: u32 = 0u; i < 4u; i++) {
        omegas[i]      = c.wheels[i * 2u].x;
        tire_temps[i]  = c.wheels[i * 2u + 1u].x;
        slip_ratios[i] = c.wheels[i * 2u].y;
        slip_angles[i] = c.wheels[i * 2u].z;
    }

    // --- A) Starter motor ---
    if (is_stalled != 0u && flag(c.config_flags,0u)) {
        if (c.current_gear <= 2u && c.throttle > 0.5) {
            starter_timer += dt_safe;
            if (starter_timer >= 1.0) {
                is_stalled    = 0u;
                starter_timer = 0.0;
                current_rpm   = 800.0;
                for (var i: u32 = 0u; i < 4u; i++) { omegas[i] = 0.0; }
            }
        } else {
            starter_timer = 0.0;
        }
    }

    // --- B) Transmission ---
    // Gear change (manual_shift_req) is managed on the CPU before the dispatch — the host
    // delivers current_gear already updated. Here we only decay the shift_timer.
    if (shift_timer > 0.0) { shift_timer -= dt_safe; }

    // --- C) RPM + Clutch ---
    var drive_omega_avg: f32;
    switch c.drivetrain {
        case 0u: { drive_omega_avg = (omegas[0] + omegas[1]) * 0.5; }
        case 1u: { drive_omega_avg = (omegas[2] + omegas[3]) * 0.5; }
        default: { drive_omega_avg = (omegas[0] + omegas[1] + omegas[2] + omegas[3]) * 0.25; }
    }

    let wheel_rpm = abs(drive_omega_avg) * total_ratio_abs * 60.0 / PI2;
    let free_rev  = 800.0 + c.throttle * (c.max_rpm - 800.0);

    var clutch_blend: f32;
    if (c.current_gear == 1u) {
        clutch_blend = 0.0;
    } else if (wheel_rpm < 1500.0) {
        clutch_blend = clamp(wheel_rpm / 1500.0, 0.0, 1.0);
    } else {
        clutch_blend = 1.0;
    }

    var raw_rpm = free_rev * (1.0 - clutch_blend) + wheel_rpm * clutch_blend;
    let clutch_slip = clutch_blend < 0.95;

    if (shift_timer > 0.0) {
        let fi = max(c.flywheel_inertia, 0.05);
        let rpm_decay = current_rpm * dt_safe / fi;
        raw_rpm = max(current_rpm - rpm_decay * 50.0, 800.0);
    } else if (flag(c.config_flags,0u) && !clutch_slip && is_stalled == 0u && raw_rpm < 550.0) {
        is_stalled    = 1u;
        starter_timer = 0.0;
    }

    // --- D) Engine thermals ---
    if (is_stalled == 0u && flag(c.config_flags,1u)) {
        let rpm_ratio = current_rpm / c.max_rpm;
        // Bit 7 opts into the coevolved organ.  Legacy states have this bit
        // clear and retain the exact historical equations.
        let organ_thermal = flag(c.config_flags,7u);
        let engine_heat_scale = select(1.0, c.max_torque / BASE_ENGINE_TORQUE_NM, organ_thermal);
        let cooling_scale = select(1.0, 1.0 + c._pad_before_wind, organ_thermal);
        let heat_gen  = rpm_ratio * c.throttle * 15.0 * engine_heat_scale * dt_safe;
        let cooling   = (speed * 0.3 + 2.0) * cooling_scale * dt_safe;
        engine_temp = clamp(engine_temp + heat_gen - cooling, 20.0, 150.0);
        oil_temp    = oil_temp + (engine_temp - oil_temp) * 0.01 * dt_safe;
    } else {
        engine_temp = max(engine_temp - 1.0 * dt_safe, 20.0);
    }

    var thermal_power = 1.0;
    if (flag(c.config_flags,1u) && engine_temp > 105.0) {
        thermal_power = clamp(1.0 - (engine_temp - 105.0) / 45.0, 0.1, 1.0);
    }

    // --- Drive torque ---
    var drive_torque:     f32;
    var drive_torque_out: f32;
    if (is_stalled != 0u) {
        current_rpm      = 0.0;
        drive_torque     = 0.0;
        drive_torque_out = 0.0;
    } else {
        current_rpm = clamp(raw_rpm, 800.0, c.max_rpm + 250.0);
        let rpm_ratio = current_rpm / c.max_rpm;

        // Realistic torque curve (peak at ~60% RPM) — physics.rs line 382
        var tm = 1.0 - (rpm_ratio - 0.6) * (rpm_ratio - 0.6) * 2.0;
        if (rpm_ratio > 1.0) {
            tm = 0.0;
        } else if (rpm_ratio < 0.2) {
            tm *= rpm_ratio / 0.2;
        }
        tm = clamp(tm, 0.0, 1.0);

        // TCS — RWD only
        var tcs_cut = 1.0;
        if (flag(c.config_flags,4u) && c.drivetrain == 1u) {
            let slip_avg = (max(slip_ratios[2], 0.0) + max(slip_ratios[3], 0.0)) * 0.5;
            if (slip_avg > 0.15) {
                tcs_cut = 1.0 - clamp((slip_avg - 0.15) * 5.0, 0.0, 0.9);
            }
        }

        let clutch_eng = select(0.0, 1.0, shift_timer <= 0.0);
        drive_torque     = c.max_torque * tm * (c.throttle * tcs_cut) * total_ratio * clutch_eng * thermal_power;
        drive_torque_out = drive_torque;
    }

    // --- E) Aerodynamics ---
    forces += c.wind * c.mass * 0.5;

    if (flag(c.config_flags,2u)) {  // enable_aero_drag
        let drag_mag = 0.5 * RHO_AR * speed * speed * c.drag_coeff * FRONTAL_AREA;
        forces += -nor0(c.vel) * drag_mag;
    }

    if (flag(c.config_flags,3u)) {  // enable_downforce
        let cl        = c.drag_coeff * 2.5;
        let downforce = 0.5 * RHO_AR * speed * speed * cl * FRONTAL_AREA;
        forces.y -= downforce;
    }

    // Rolling resistance
    let total_load_rr = normal_loads[0] + normal_loads[1] + normal_loads[2] + normal_loads[3];
    if (speed > 0.05) {
        forces += -nor0(c.vel) * (0.015 * total_load_rr);
    }

    // Engine braking (same as CPU line 432-438)
    if (is_stalled == 0u && c.throttle < 0.05 && abs(forward_vel) > 1.0) {
        let eb = total_ratio_abs * 80.0;
        let max_eb = abs(forward_vel) * c.mass / dt_safe;
        forces += -forward_dir * sign(forward_vel) * min(eb, max_eb);
    }

    // --- F) Terrain and environment grip ---
    let mu_base = c.mu_peak * c.terrain_grip * c.env_grip;

    // --- G) Tyres per wheel ---
    let wheel_long_pos = array<f32, 4>(half_wb, half_wb, -half_wb, -half_wb);
    let wheel_lat_pos  = array<f32, 4>(-half_tw, half_tw, -half_tw, half_tw);

    // LSD / torque distribution
    var wheel_torques = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    switch c.drivetrain {
        case 0u: {  // FWD — uniform split
            wheel_torques[0] = drive_torque * 0.5;
            wheel_torques[1] = drive_torque * 0.5;
        }
        case 1u: {  // RWD — LSD
            let delta_omega  = abs(omegas[3] - omegas[2]);
            let lock_factor  = clamp(c.lsd_lock_ratio * delta_omega * 0.5, 0.0, 0.4);
            // POSSIBLE CPU BUG replicated: split_left not clamped to [0,1]
            let split_left   = 0.5 + lock_factor * sign(omegas[3] - omegas[2]);
            wheel_torques[2] = drive_torque * split_left;
            wheel_torques[3] = drive_torque * (1.0 - split_left);
        }
        default: {  // AWD
            wheel_torques[0] = drive_torque * 0.25;
            wheel_torques[1] = drive_torque * 0.25;
            wheel_torques[2] = drive_torque * 0.25;
            wheel_torques[3] = drive_torque * 0.25;
        }
    }

    // Brake bias: 30/30/20/20
    let brake_total = c.brake_cmd * c.brake_torque_max;
    let brake_pw = array<f32, 4>(
        brake_total * 0.30,
        brake_total * 0.30,
        brake_total * 0.20,
        brake_total * 0.20,
    );

    var total_long_force = 0.0;
    var total_lat_force  = 0.0;
    var total_yaw_torque = 0.0;

    // Speed-sensitive steering (CPU line 518-524)
    var max_steer_deg: f32;
    if (speed < 15.0) {
        max_steer_deg = 35.0;
    } else {
        max_steer_deg = max(35.0 - (speed - 15.0) * 0.4, 12.0);
    }
    let max_steer_rad = max_steer_deg * 0.017453292519943295;

    let lsd_speed_gate = clamp(speed / 5.0, 0.0, 1.0);

    for (var i: u32 = 0u; i < 4u; i++) {
        let load = max(normal_loads[i], 10.0);

        // Tyre temperature
        var temp_factor = 1.0;
        if (flag(c.config_flags,1u)) {
            temp_factor = clamp(1.0 - abs(tire_temps[i] - c.optimal_temp) * 0.5, 0.7, 1.0);
        }
        let mu = mu_base * temp_factor;

        let wheel_linear_speed = omegas[i] * c.wheel_radius;

        // Velocity at the contact point (includes angular yaw — physics.rs lines 514-515)
        let raw_fwd = forward_vel + c.av_y * wheel_lat_pos[i];
        let raw_lat = lateral_vel - c.av_y * wheel_long_pos[i];

        // Steering angle (front wheels only: i < 2)
        let wheel_steer = select(0.0, c.steer * max_steer_rad, i < 2u);
        let cos_s = cos(wheel_steer);
        let sin_s = sin(wheel_steer);

        // Velocity in the wheel's reference frame
        let contact_fwd = raw_fwd * cos_s + raw_lat * sin_s;
        let contact_lat = -raw_fwd * sin_s + raw_lat * cos_s;

        // Slip ratio (longitudinal Pacejka)
        // POSSIBLE DIVERGENCE: max() and clamp() are deterministic; IEEE-754 division. ±1 ULP.
        let v_ref = max(abs(wheel_linear_speed), max(abs(contact_fwd), 0.5));
        let sr    = clamp((wheel_linear_speed - contact_fwd) / v_ref, -1.0, 1.0);

        // Longitudinal Pacejka: F = load * mu * sin(c_long * atan(b_long * sr))
        // POSSIBLE DIVERGENCE: GPU atan may differ from CPU atanf by ±1 ULP.
        let b_long: f32 = 10.0;
        let c_long: f32 = 1.5;
        let f_long = load * mu * sin(c_long * atan(b_long * sr));

        // Lateral Pacejka
        var sa:    f32 = 0.0;
        var f_lat: f32;
        if (abs(contact_fwd) > 0.5) {
            // POSSIBLE DIVERGENCE: GPU atan2 vs CPU atan2f ±1 ULP.
            sa    = atan2(contact_lat, abs(contact_fwd));
            let b_lat: f32 = 12.0;
            let c_lat: f32 = 1.35;
            f_lat = load * mu * sin(c_lat * atan(b_lat * sa));
        } else {
            // Ice-bug fix: viscous synthetic grip (CPU line 553)
            f_lat = clamp(contact_lat * load * mu * 2.0, -load * mu, load * mu);
        }

        // Kamm circle
        // POSSIBLE DIVERGENCE: GPU length() vs CPU hypotf ±1 ULP.
        let max_force = mu * load;
        let combined  = length(vec2<f32>(f_long, f_lat));
        var f_long_f: f32;
        var f_lat_f:  f32;
        if (combined > max_force && combined > 0.001) {
            let scale = max_force / combined;
            f_long_f = f_long * scale;
            f_lat_f  = f_lat  * scale;
        } else {
            f_long_f = f_long;
            f_lat_f  = f_lat;
        }

        // Wheel omega integration
        let tire_react = f_long_f * c.wheel_radius;
        let base_t     = wheel_torques[i] - tire_react;
        var new_omega  = omegas[i] + (base_t / c.wheel_inertia) * dt_safe;

        let brk = brake_pw[i];
        if (brk > 0.0) {
            let stop_t = abs(new_omega) * c.wheel_inertia / dt_safe;
            var actual_brk: f32;
            if (flag(c.config_flags,5u)) {  // enable_abs
                actual_brk = min(brk, stop_t);
            } else {
                actual_brk = brk;
            }
            if (new_omega > 0.0) {
                new_omega -= (actual_brk / c.wheel_inertia) * dt_safe;
            } else if (new_omega < 0.0) {
                new_omega += (actual_brk / c.wheel_inertia) * dt_safe;
            }
        }
        omegas[i] = new_omega;

        // Aquaplaning (boolean flag passed as 0.1 or 1.0 by the host)
        let aqua = c.aquaplaning;

        // Rotation back to the chassis reference frame
        let ch_long = f_long_f * cos_s - f_lat_f * sin_s;
        let ch_lat  = f_long_f * sin_s + f_lat_f * cos_s;

        total_long_force += ch_long * aqua;
        total_lat_force  += ch_lat  * aqua;
        total_yaw_torque += ch_lat  * wheel_long_pos[i] * aqua;
        total_yaw_torque += ch_long * wheel_lat_pos[i]  * 0.005 * lsd_speed_gate;

        // Tyre thermals
        if (flag(c.config_flags,1u)) {
            let slip_heat  = (abs(sr) + abs(sa)) * load * 0.000002 * dt_safe;
            // NOTE: brake_heat uses new_omega (post-braking), same as the CPU which uses the
            // value updated in the same iteration (writeback happens at the end of the CPU loop).
            let brake_heat = brake_pw[i] * abs(new_omega) * 0.000001 * dt_safe;
            let cooling    = 0.02 * (1.0 + speed * 0.01) * dt_safe;
            tire_temps[i]  = clamp(tire_temps[i] + slip_heat + brake_heat - cooling, 0.0, 1.0);
        }

        slip_ratios[i] = sr;
        slip_angles[i] = sa;
    }

    // Apply tyre forces to the chassis
    forces  += forward_dir * total_long_force;
    forces  += -lateral_dir * total_lat_force;
    torques_y += total_yaw_torque;

    // yaw_inertia: the CPU's formula (dead assumption documented in the header)
    let yaw_inertia = c.mass * (half_wb * half_wb);

    // ============================================================
    // === PHASE 4: WALL COLLISION                              ===
    // ============================================================
    // The CPU (collision_system) modifies body.velocity and body.angular_velocity.y BEFORE
    // integration_system. To replicate: we use vel_col (starts at c.vel, pre-integration)
    // and av_y_col (starts at c.av_y). Phase 5 integration starts from these variables.
    var vel_col  = c.vel;
    var pos_col  = c.pos;
    var av_y_col = c.av_y;

    var probe_pts: array<vec3<f32>, 4>;
    probe_pts[0] = c.pos + fwd * half_wb;
    probe_pts[1] = c.pos - fwd * half_wb;
    probe_pts[2] = c.pos + rgt * half_tw;
    probe_pts[3] = c.pos - rgt * half_tw;

    for (var p: u32 = 0u; p < 4u; p++) {
        let probe = probe_pts[p];
        if (voxel_at(probe.x, probe.z) == 2u) {  // Wall
            let pen_dir  = nor0(probe - c.pos);
            let vel_into = dot(vel_col, pen_dir);
            if (vel_into > 0.0) {
                // Impulse with restitution 0.05 (same as CPU line 677)
                let impulse = -(1.0 + 0.05) * vel_into * c.mass;
                vel_col += pen_dir * (impulse / c.mass);
                // Tangential friction (CPU line 682-683)
                let tang = vel_col - pen_dir * dot(vel_col, pen_dir);
                vel_col -= tang * 0.5;
                // Yaw kick: ADDED DIRECTLY to av_y — same as CPU (line 687)
                // NOT a torque; ignores yaw_inertia here (the CPU does not divide either).
                let yaw_kick = cross(fwd, pen_dir).y;
                av_y_col += yaw_kick * vel_into * 0.05;
                // Push out of the wall
                pos_col -= pen_dir * 0.5;
            }
        }
    }

    // Ground clamp with heightmap (CPU lines 696-700)
    let ground_y_c = height_at(c.pos.x, c.pos.z) + 0.5;
    if (pos_col.y < ground_y_c) {
        pos_col.y = ground_y_c;
        vel_col.y = max(vel_col.y, 0.0);
    }

    // ============================================================
    // === PHASE 5: INTEGRATION (Semi-Implicit Euler)           ===
    // ============================================================
    // The CPU starts from vel_col (post-collision). Reapplies aero drag (DUPLICATION — see header).
    let speed_int = length(vel_col);
    var drag_force_int = vec3<f32>(0.0, 0.0, 0.0);
    if (flag(c.config_flags,2u)) {
        let drag_mag = 0.5 * RHO_AR * c.drag_coeff * FRONTAL_AREA * speed_int * speed_int;
        drag_force_int = -nor0(vel_col) * drag_mag;
    }

    let accel    = (forces + drag_force_int) / c.mass;
    var final_vel = vel_col + accel * dt_safe;

    // Lateral scrub friction (CPU lines 727-735)
    let fwd_spd    = dot(final_vel, forward_dir);
    let lat_v      = final_vel - forward_dir * fwd_spd;
    if (length(lat_v) > 0.5) {
        let scrub = max(1.0 - 3.0 * dt_safe, 0.85);
        final_vel = forward_dir * fwd_spd + lat_v * scrub;
    }

    var final_pos = pos_col + final_vel * dt_safe;

    // Minimum ground constraint (CPU line 754-756)
    if (final_pos.y < 0.5) {
        final_pos.y = 0.5;
        final_vel.y = max(final_vel.y, 0.0);
    }

    // Yaw: damping scales with speed (CPU v10 — 400 + speed * 60)
    let base_damp  = 400.0;
    let speed_damp = speed_int * 60.0;
    // av_y_col includes direct collision kicks (already applied before integration, as on the CPU)
    let drag_y  = -av_y_col * (base_damp + speed_damp);
    let yi      = select(c.mass * 2.0, yaw_inertia, yaw_inertia > 0.0);
    let ang_a_y = (torques_y + drag_y) / yi;
    let final_av_y = av_y_col + ang_a_y * dt_safe;

    // Yaw rotation: the host applies transform.rotate_y(final_av_y * dt) after reading the output.

    // ============================================================
    // === OUTPUT WRITE                                         ===
    // ============================================================
    var out: CarOutput;
    out.vel          = final_vel;
    out.av_y         = final_av_y;
    out.pos          = final_pos;
    out._pad_pos     = 0.0;
    out.fwd          = fwd;
    out._pad_fwd     = 0.0;
    out.lat          = c.lat;
    out._pad_lat     = 0.0;
    out.current_rpm  = current_rpm;
    out.drive_torque_out = drive_torque_out;
    out.shift_timer  = max(shift_timer, 0.0);
    out.is_stalled   = is_stalled;
    out.starter_timer = starter_timer;
    out.engine_temp  = engine_temp;
    out.oil_temp     = oil_temp;
    out._pad_eng     = 0.0;

    for (var i: u32 = 0u; i < 4u; i++) {
        out.wheels[i * 2u]      = vec4<f32>(omegas[i], slip_ratios[i], slip_angles[i], normal_loads[i]);
        out.wheels[i * 2u + 1u] = vec4<f32>(tire_temps[i], susp_comps[i], prev_comps[i], 0.0);
    }

    output[being] = out;
}
