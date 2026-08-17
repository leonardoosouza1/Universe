// Bisection phase: suspension + PHASE 3 (powertrain + tires + aero).
// The most complex phase: switch, loops with Pacejka (atan/sin), large arrays.

struct CarInput {
    vel:        vec3<f32>,
    av_y:       f32,
    wheels:     array<vec4<f32>, 8>,
    pos:        vec3<f32>,
    _pad_pos:   f32,
    fwd:        vec3<f32>,
    _pad_fwd:   f32,
    lat:        vec3<f32>,
    _pad_lat:   f32,
    max_torque:      f32,
    current_rpm:     f32,
    max_rpm:         f32,
    throttle:        f32,
    brake_cmd:       f32,
    steer:           f32,
    drive_torque_out:f32,
    current_gear:    u32,
    shift_timer:     f32,
    is_stalled:      u32,
    starter_timer:   f32,
    flywheel_inertia:f32,
    engine_temp:     f32,
    oil_temp:        f32,
    mass:            f32,
    drag_coeff:      f32,
    wheel_base:      f32,
    cg_height:       f32,
    track_width:     f32,
    lsd_lock_ratio:  f32,
    brake_torque_max:f32,
    drivetrain:      u32,
    mu_peak:             f32,
    cornering_stiffness: f32,
    wheel_radius:        f32,
    wheel_inertia:       f32,
    optimal_temp:        f32,
    susp_stiffness:  f32,
    susp_damping:    f32,
    susp_rest:       f32,
    susp_max_travel: f32,
    _pad_before_wind: f32,
    wind:            vec3<f32>,
    _pad_wind:       f32,
    terrain_grip:    f32,
    env_grip:        f32,
    aquaplaning:     f32,
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
@group(0) @binding(1) var<storage, read>       grid:    array<u32>;
@group(0) @binding(2) var<storage, read>       hmap:    array<f32>;
@group(0) @binding(3) var<storage, read>       input:   array<CarInput>;
@group(0) @binding(4) var<storage, read_write> output:  array<CarOutput>;

const GRAVITY:      f32 = 9.81;
const RHO_AR:       f32 = 1.225;
const FRONTAL_AREA: f32 = 2.2;
const PI2:          f32 = 6.283185307179586;
const FINAL_DRIVE:  f32 = 3.7;

fn flag(flags: u32, bit: u32) -> bool {
    return (flags >> bit & 1u) != 0u;
}

fn nor0(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l < 1e-9) { return vec3<f32>(0.0); }
    return v / l;
}

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
    return (h00*(1.0-tx)+h10*tx)*(1.0-tz) + (h01*(1.0-tx)+h11*tx)*tz;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let being = gid.x;
    if (being >= params.n_beings) { return; }

    let c = input[being];
    let dt_safe = max(c.dt, 0.001);
    let speed = length(c.vel);

    // --- Suspension (phase 2) ---
    let aero_downforce = select(0.0, 3.0 * speed * speed, flag(c.config_flags, 3u));
    let dpw = aero_downforce * 0.25;
    let static_load = c.mass * GRAVITY / 4.0;
    let fwd = c.fwd; let rgt = -c.lat;
    let half_wb = c.wheel_base * 0.5; let half_tw = c.track_width * 0.5;
    var wheel_offsets: array<vec3<f32>, 4>;
    wheel_offsets[0] = fwd * half_wb - rgt * half_tw;
    wheel_offsets[1] = fwd * half_wb + rgt * half_tw;
    wheel_offsets[2] = -fwd * half_wb - rgt * half_tw;
    wheel_offsets[3] = -fwd * half_wb + rgt * half_tw;
    var normal_loads: array<f32, 4>;
    var susp_comps:   array<f32, 4>;
    var prev_comps:   array<f32, 4>;
    var total_susp_load = 0.0;
    for (var i: u32 = 0u; i < 4u; i++) {
        let wp = c.pos + wheel_offsets[i];
        let gy = height_at(wp.x, wp.z);
        let comp = c.susp_rest - (wp.y - gy);
        let prev_c = c.wheels[i * 2u + 1u].z;
        if (comp > 0.0 && comp < c.susp_max_travel) {
            let sf = c.susp_stiffness * comp;
            let df = c.susp_damping * (comp - prev_c) / dt_safe;
            let tf = max(sf + df, 0.0) + dpw;
            normal_loads[i] = clamp(tf, static_load * 0.2, static_load * 6.0);
            susp_comps[i]   = comp;
        } else {
            normal_loads[i] = 0.0;
            susp_comps[i]   = 0.0;
        }
        prev_comps[i] = comp;
        total_susp_load += normal_loads[i];
    }
    if (total_susp_load < 100.0) {
        let pw = c.mass * GRAVITY * 0.25;
        for (var i: u32 = 0u; i < 4u; i++) { normal_loads[i] = pw; }
    }

    // --- Phase 3: powertrain + tires ---
    let forward_dir = fwd;
    let lateral_dir = c.lat;
    let forward_vel = dot(c.vel, forward_dir);
    let lateral_vel = dot(c.vel, lateral_dir);

    let gratio = gear_ratio(c.current_gear);
    let total_ratio = gratio * FINAL_DRIVE;
    let total_ratio_abs = abs(total_ratio);

    var is_stalled    = c.is_stalled;
    var starter_timer = c.starter_timer;
    var shift_timer   = c.shift_timer;
    var current_rpm   = c.current_rpm;
    var engine_temp   = c.engine_temp;
    var oil_temp      = c.oil_temp;

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

    if (is_stalled != 0u && flag(c.config_flags, 0u)) {
        if (c.current_gear <= 2u && c.throttle > 0.5) {
            starter_timer += dt_safe;
            if (starter_timer >= 1.0) {
                is_stalled = 0u; starter_timer = 0.0; current_rpm = 800.0;
                for (var i: u32 = 0u; i < 4u; i++) { omegas[i] = 0.0; }
            }
        } else { starter_timer = 0.0; }
    }
    if (shift_timer > 0.0) { shift_timer -= dt_safe; }

    var drive_omega_avg: f32;
    switch c.drivetrain {
        case 0u: { drive_omega_avg = (omegas[0] + omegas[1]) * 0.5; }
        case 1u: { drive_omega_avg = (omegas[2] + omegas[3]) * 0.5; }
        default: { drive_omega_avg = (omegas[0] + omegas[1] + omegas[2] + omegas[3]) * 0.25; }
    }
    let wheel_rpm = abs(drive_omega_avg) * total_ratio_abs * 60.0 / PI2;
    let free_rev  = 800.0 + c.throttle * (c.max_rpm - 800.0);
    var clutch_blend: f32;
    if (c.current_gear == 1u) { clutch_blend = 0.0; }
    else if (wheel_rpm < 1500.0) { clutch_blend = clamp(wheel_rpm / 1500.0, 0.0, 1.0); }
    else { clutch_blend = 1.0; }
    var raw_rpm = free_rev * (1.0 - clutch_blend) + wheel_rpm * clutch_blend;
    let clutch_slip = clutch_blend < 0.95;
    if (shift_timer > 0.0) {
        let fi = max(c.flywheel_inertia, 0.05);
        raw_rpm = max(current_rpm - current_rpm * dt_safe / fi * 50.0, 800.0);
    } else if (flag(c.config_flags, 0u) && !clutch_slip && is_stalled == 0u && raw_rpm < 550.0) {
        is_stalled = 1u; starter_timer = 0.0;
    }
    if (is_stalled == 0u && flag(c.config_flags, 1u)) {
        let rr = current_rpm / c.max_rpm;
        let heat_gen = rr * c.throttle * 15.0 * dt_safe;
        let cooling  = (speed * 0.3 + 2.0) * dt_safe;
        engine_temp = clamp(engine_temp + heat_gen - cooling, 20.0, 150.0);
        oil_temp = oil_temp + (engine_temp - oil_temp) * 0.01 * dt_safe;
    } else { engine_temp = max(engine_temp - 1.0 * dt_safe, 20.0); }
    var thermal_power = 1.0;
    if (flag(c.config_flags, 1u) && engine_temp > 105.0) {
        thermal_power = clamp(1.0 - (engine_temp - 105.0) / 45.0, 0.1, 1.0);
    }
    var drive_torque: f32; var drive_torque_out: f32;
    if (is_stalled != 0u) {
        current_rpm = 0.0; drive_torque = 0.0; drive_torque_out = 0.0;
    } else {
        current_rpm = clamp(raw_rpm, 800.0, c.max_rpm + 250.0);
        let rr = current_rpm / c.max_rpm;
        var tm = 1.0 - (rr - 0.6) * (rr - 0.6) * 2.0;
        if (rr > 1.0) { tm = 0.0; } else if (rr < 0.2) { tm *= rr / 0.2; }
        tm = clamp(tm, 0.0, 1.0);
        var tcs_cut = 1.0;
        if (flag(c.config_flags, 4u) && c.drivetrain == 1u) {
            let sa = (max(slip_ratios[2], 0.0) + max(slip_ratios[3], 0.0)) * 0.5;
            if (sa > 0.15) { tcs_cut = 1.0 - clamp((sa - 0.15) * 5.0, 0.0, 0.9); }
        }
        let clutch_eng = select(0.0, 1.0, shift_timer <= 0.0);
        drive_torque = c.max_torque * tm * c.throttle * tcs_cut * total_ratio * clutch_eng * thermal_power;
        drive_torque_out = drive_torque;
    }

    var forces = vec3<f32>(0.0, -GRAVITY * c.mass, 0.0);
    forces += c.wind * c.mass * 0.5;
    if (flag(c.config_flags, 2u)) {
        let drag_mag = 0.5 * RHO_AR * speed * speed * c.drag_coeff * FRONTAL_AREA;
        forces += -nor0(c.vel) * drag_mag;
    }
    if (flag(c.config_flags, 3u)) {
        let cl = c.drag_coeff * 2.5;
        forces.y -= 0.5 * RHO_AR * speed * speed * cl * FRONTAL_AREA;
    }
    let total_load_rr = normal_loads[0]+normal_loads[1]+normal_loads[2]+normal_loads[3];
    if (speed > 0.05) { forces += -nor0(c.vel) * (0.015 * total_load_rr); }
    if (is_stalled == 0u && c.throttle < 0.05 && abs(forward_vel) > 1.0) {
        let eb = total_ratio_abs * 80.0;
        let max_eb = abs(forward_vel) * c.mass / dt_safe;
        forces += -forward_dir * sign(forward_vel) * min(eb, max_eb);
    }

    let mu_base = c.mu_peak * c.terrain_grip * c.env_grip;
    let wheel_long_pos = array<f32, 4>(half_wb, half_wb, -half_wb, -half_wb);
    let wheel_lat_pos  = array<f32, 4>(-half_tw, half_tw, -half_tw, half_tw);
    var wheel_torques = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    switch c.drivetrain {
        case 0u: { wheel_torques[0] = drive_torque * 0.5; wheel_torques[1] = drive_torque * 0.5; }
        case 1u: {
            let dw = abs(omegas[3]-omegas[2]);
            let lf = clamp(c.lsd_lock_ratio*dw*0.5, 0.0, 0.4);
            let sl = 0.5 + lf * sign(omegas[3]-omegas[2]);
            wheel_torques[2] = drive_torque * sl; wheel_torques[3] = drive_torque * (1.0-sl);
        }
        default: { for (var i: u32=0u;i<4u;i++){wheel_torques[i]=drive_torque*0.25;} }
    }
    let brake_total = c.brake_cmd * c.brake_torque_max;
    let brake_pw = array<f32, 4>(brake_total*0.30, brake_total*0.30, brake_total*0.20, brake_total*0.20);
    var total_long_force = 0.0; var total_lat_force = 0.0; var total_yaw_torque = 0.0;
    var max_steer_deg: f32;
    if (speed < 15.0) { max_steer_deg = 35.0; }
    else { max_steer_deg = max(35.0-(speed-15.0)*0.4, 12.0); }
    let max_steer_rad = max_steer_deg * 0.017453292519943295;
    let lsd_speed_gate = clamp(speed / 5.0, 0.0, 1.0);

    for (var i: u32 = 0u; i < 4u; i++) {
        let load = max(normal_loads[i], 10.0);
        var temp_factor = 1.0;
        if (flag(c.config_flags, 1u)) {
            temp_factor = clamp(1.0 - abs(tire_temps[i]-c.optimal_temp)*0.5, 0.7, 1.0);
        }
        let mu = mu_base * temp_factor;
        let wheel_linear_speed = omegas[i] * c.wheel_radius;
        let raw_fwd = forward_vel + c.av_y * wheel_lat_pos[i];
        let raw_lat = lateral_vel - c.av_y * wheel_long_pos[i];
        let wheel_steer = select(0.0, c.steer * max_steer_rad, i < 2u);
        let cos_s = cos(wheel_steer); let sin_s = sin(wheel_steer);
        let contact_fwd = raw_fwd * cos_s + raw_lat * sin_s;
        let contact_lat = -raw_fwd * sin_s + raw_lat * cos_s;
        let v_ref = max(abs(wheel_linear_speed), max(abs(contact_fwd), 0.5));
        let sr = clamp((wheel_linear_speed - contact_fwd) / v_ref, -1.0, 1.0);
        let f_long = load * mu * sin(1.5 * atan(10.0 * sr));
        var sa: f32 = 0.0; var f_lat: f32;
        if (abs(contact_fwd) > 0.5) {
            sa = atan2(contact_lat, abs(contact_fwd));
            f_lat = load * mu * sin(1.35 * atan(12.0 * sa));
        } else {
            f_lat = clamp(contact_lat * load * mu * 2.0, -load * mu, load * mu);
        }
        let max_force = mu * load;
        let combined  = length(vec2<f32>(f_long, f_lat));
        var f_long_f: f32; var f_lat_f: f32;
        if (combined > max_force && combined > 0.001) {
            let scale = max_force / combined;
            f_long_f = f_long * scale; f_lat_f = f_lat * scale;
        } else { f_long_f = f_long; f_lat_f = f_lat; }
        let tire_react = f_long_f * c.wheel_radius;
        let base_t = wheel_torques[i] - tire_react;
        var new_omega = omegas[i] + (base_t / c.wheel_inertia) * dt_safe;
        let brk = brake_pw[i];
        if (brk > 0.0) {
            let stop_t = abs(new_omega) * c.wheel_inertia / dt_safe;
            var actual_brk: f32;
            if (flag(c.config_flags, 5u)) { actual_brk = min(brk, stop_t); }
            else { actual_brk = brk; }
            if (new_omega > 0.0) { new_omega -= (actual_brk/c.wheel_inertia)*dt_safe; }
            else if (new_omega < 0.0) { new_omega += (actual_brk/c.wheel_inertia)*dt_safe; }
        }
        omegas[i] = new_omega;
        let aqua = c.aquaplaning;
        let ch_long = f_long_f * cos_s - f_lat_f * sin_s;
        let ch_lat  = f_long_f * sin_s + f_lat_f * cos_s;
        total_long_force += ch_long * aqua;
        total_lat_force  += ch_lat  * aqua;
        total_yaw_torque += ch_lat * wheel_long_pos[i] * aqua;
        total_yaw_torque += ch_long * wheel_lat_pos[i] * 0.005 * lsd_speed_gate;
        if (flag(c.config_flags, 1u)) {
            let slip_heat  = (abs(sr) + abs(sa)) * load * 0.000002 * dt_safe;
            let brake_heat = brake_pw[i] * abs(new_omega) * 0.000001 * dt_safe;
            let cooling    = 0.02 * (1.0 + speed * 0.01) * dt_safe;
            tire_temps[i]  = clamp(tire_temps[i] + slip_heat + brake_heat - cooling, 0.0, 1.0);
        }
        slip_ratios[i] = sr;
        slip_angles[i] = sa;
    }
    forces  += forward_dir * total_long_force;
    forces  -= lateral_dir * total_lat_force;

    var out: CarOutput;
    out.vel          = forces / c.mass; // resultant force as proxy
    out.av_y         = total_yaw_torque;
    out.pos          = c.pos;
    out._pad_pos     = 0.0;
    out.fwd          = fwd;
    out._pad_fwd     = 0.0;
    out.lat          = c.lat;
    out._pad_lat     = 0.0;
    out.current_rpm  = current_rpm;
    out.drive_torque_out = drive_torque_out;
    out.shift_timer  = shift_timer;
    out.is_stalled   = is_stalled;
    out.starter_timer = starter_timer;
    out.engine_temp  = engine_temp;
    out.oil_temp     = oil_temp;
    out._pad_eng     = 0.0;
    for (var i: u32 = 0u; i < 4u; i++) {
        out.wheels[i*2u]   = vec4<f32>(omegas[i], slip_ratios[i], slip_angles[i], normal_loads[i]);
        out.wheels[i*2u+1u] = vec4<f32>(tire_temps[i], susp_comps[i], prev_comps[i], 0.0);
    }
    output[being] = out;
}
