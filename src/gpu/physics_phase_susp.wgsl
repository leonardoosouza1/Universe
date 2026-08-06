// Fase de bissecção: shader mínimo + FASE 2 (suspensão + carga normal).
// Inclui: arrays var<private>, loops, acesso ao heightmap, downforce.
// Não inclui: powertrain, pneus, colisão, integração.

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

const GRAVITY: f32 = 9.81;

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

fn flag(flags: u32, bit: u32) -> bool {
    return (flags >> bit & 1u) != 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let being = gid.x;
    if (being >= params.n_beings) { return; }

    let c = input[being];
    let dt_safe = max(c.dt, 0.001);

    let speed = length(c.vel);
    let aero_downforce = select(0.0, 3.0 * speed * speed, flag(c.config_flags, 3u));
    let downforce_per_wheel = aero_downforce * 0.25;
    let static_load = c.mass * GRAVITY / 4.0;

    let fwd = c.fwd;
    let rgt = -c.lat;
    let half_wb = c.wheel_base * 0.5;
    let half_tw = c.track_width * 0.5;

    // Arrays locais var<private> — esta é a parte suspeita
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
        let wheel_pos = c.pos + wheel_offsets[i];
        let ground_y  = height_at(wheel_pos.x, wheel_pos.z);
        let height_above = wheel_pos.y - ground_y;
        let compression  = c.susp_rest - height_above;
        let prev_c = c.wheels[i * 2u + 1u].z;

        if (compression > 0.0 && compression < c.susp_max_travel) {
            let spring_f = c.susp_stiffness * compression;
            let comp_vel = (compression - prev_c) / dt_safe;
            let damper_f = c.susp_damping * comp_vel;
            let total_f  = max(spring_f + damper_f, 0.0) + downforce_per_wheel;
            let clamped  = clamp(total_f, static_load * 0.2, static_load * 6.0);
            normal_loads[i] = clamped;
            susp_comps[i]   = compression;
        } else {
            normal_loads[i] = 0.0;
            susp_comps[i]   = 0.0;
        }
        prev_comps[i]    = compression;
        total_susp_load += normal_loads[i];
    }

    if (total_susp_load < 100.0) {
        let pw = c.mass * GRAVITY * 0.25;
        for (var i: u32 = 0u; i < 4u; i++) {
            normal_loads[i] = pw;
        }
    }

    // Saída: usa normal_loads[0] como proxy no av_y para verificar que o cálculo rodou
    var out: CarOutput;
    out.vel          = c.vel;
    out.av_y         = normal_loads[0]; // proxy para verificar resultado da suspensão
    out.pos          = c.pos;
    out._pad_pos     = 0.0;
    out.fwd          = c.fwd;
    out._pad_fwd     = 0.0;
    out.lat          = c.lat;
    out._pad_lat     = 0.0;
    out.current_rpm  = c.current_rpm;
    out.drive_torque_out = 0.0;
    out.shift_timer  = susp_comps[0]; // proxy
    out.is_stalled   = c.is_stalled;
    out.starter_timer = 0.0;
    out.engine_temp  = c.engine_temp;
    out.oil_temp     = c.oil_temp;
    out._pad_eng     = 0.0;

    for (var i: u32 = 0u; i < 8u; i++) {
        out.wheels[i] = c.wheels[i];
    }
    // Guarda normal_load nas rodas
    for (var i: u32 = 0u; i < 4u; i++) {
        out.wheels[i * 2u].w = normal_loads[i];
        out.wheels[i * 2u + 1u].y = susp_comps[i];
        out.wheels[i * 2u + 1u].z = prev_comps[i];
    }

    output[being] = out;
}
