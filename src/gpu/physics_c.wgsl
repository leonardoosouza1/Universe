// Kernel C: wall collision + Euler integration + final write.
// Reads: original input + MidState (from A) + MidState2 (from B) → writes CarOutput.
//
// This kernel is deliberately light: no numerically intensive loops, only simple
// linear algebra + 4 voxel lookups + 1 heightmap lookup.

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

// Kernel A's buffer — we need: susp_comps, prev_comps, yaw_inertia, engine state
struct MidState {
    normal_loads:    vec4<f32>,
    susp_comps:      vec4<f32>,
    prev_comps:      vec4<f32>,
    current_rpm:     f32,
    drive_torque:    f32,
    drive_torque_out:f32,
    shift_timer:     f32,
    is_stalled:      u32,
    starter_timer:   f32,
    engine_temp:     f32,
    oil_temp:        f32,
    omegas:          vec4<f32>,
    forces:          vec3<f32>,
    _pad_f:          f32,
    torques_y:       f32,
    yaw_inertia:     f32,
    _pad2:           vec2<f32>,
}

// Kernel B's buffer — tyre forces + omegas + slip
struct MidState2 {
    forces:      vec3<f32>,
    torques_y:   f32,
    omegas:      vec4<f32>,
    slip_r:      vec4<f32>,
    slip_a:      vec4<f32>,
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
    // D5: climate mirror (uniform layout; c does not consume it).
    clima_mu:   f32,
    vento_x:    f32,
    vento_z:    f32,
    clima_freq: f32,
}

@group(0) @binding(0) var<uniform>             params:  WorldParams;
@group(0) @binding(1) var<storage, read>       grid:    array<u32>;
@group(0) @binding(2) var<storage, read>       hmap:    array<f32>;
@group(0) @binding(3) var<storage, read>       input:   array<CarInput>;
@group(0) @binding(4) var<storage, read>       mid:     array<MidState>;
@group(0) @binding(5) var<storage, read>       mid2:    array<MidState2>;
@group(0) @binding(6) var<storage, read_write> output:  array<CarOutput>;

const RHO_AR:       f32 = 1.225;
const FRONTAL_AREA: f32 = 2.2;

fn voxel_at(wx: f32, wz: f32) -> u32 {
    let x = i32(round(wx / params.cell_size));
    let z = i32(round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if (x>=0 && x<n && z>=0 && z<n) { return grid[u32(z*n+x)]; }
    return 2u;
}

fn height_at(wx: f32, wz: f32) -> f32 {
    let fx=wx/params.cell_size; let fz=wz/params.cell_size;
    let x0=i32(floor(fx)); let z0=i32(floor(fz));
    let tx=fx-f32(x0); let tz=fz-f32(z0);
    let n=i32(params.grid_size);
    var h00=0.0;var h10=0.0;var h01=0.0;var h11=0.0;
    if (x0>=0&&x0<n&&z0>=0&&z0<n)                 { h00=hmap[u32(z0*n+x0)]; }
    if ((x0+1)>=0&&(x0+1)<n&&z0>=0&&z0<n)         { h10=hmap[u32(z0*n+(x0+1))]; }
    if (x0>=0&&x0<n&&(z0+1)>=0&&(z0+1)<n)         { h01=hmap[u32((z0+1)*n+x0)]; }
    if ((x0+1)>=0&&(x0+1)<n&&(z0+1)>=0&&(z0+1)<n) { h11=hmap[u32((z0+1)*n+(x0+1))]; }
    return (h00*(1.0-tx)+h10*tx)*(1.0-tz)+(h01*(1.0-tx)+h11*tx)*tz;
}

fn nor0(v: vec3<f32>) -> vec3<f32> {
    let l=length(v); if (l<1e-9){return vec3<f32>(0.0);} return v/l;
}

fn flag(flags: u32, bit: u32) -> bool { return (flags>>bit&1u)!=0u; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let being = gid.x;
    if (being >= params.n_beings) { return; }

    let c  = input[being];
    let m  = mid[being];
    let m2 = mid2[being];
    let dt_safe = max(c.dt, 0.001);

    let fwd = c.fwd;
    let rgt = -c.lat;
    let hwb = c.wheel_base * 0.5;
    let htw = c.track_width * 0.5;

    // === PHASE 4: wall collision ===
    // m2.forces already includes the tyre forces accumulated by Kernel B
    var vel_col  = c.vel;
    var pos_col  = c.pos;
    var av_y_col = c.av_y;

    // 4 probe points — identical to the original physics.wgsl (lines 653-658)
    let p0 = c.pos + fwd * hwb;
    let p1 = c.pos - fwd * hwb;
    let p2 = c.pos + rgt * htw;
    let p3 = c.pos - rgt * htw;

    // Probe 0
    if (voxel_at(p0.x, p0.z) == 2u) {
        let pd = nor0(p0 - c.pos);
        let vi = dot(vel_col, pd);
        if (vi > 0.0) {
            vel_col += pd * (-(1.0+0.05)*vi);
            let tang = vel_col - pd*dot(vel_col,pd);
            vel_col -= tang*0.5;
            av_y_col += cross(fwd, pd).y * vi * 0.05;
            pos_col  -= pd * 0.5;
        }
    }
    // Probe 1
    if (voxel_at(p1.x, p1.z) == 2u) {
        let pd = nor0(p1 - c.pos);
        let vi = dot(vel_col, pd);
        if (vi > 0.0) {
            vel_col += pd * (-(1.0+0.05)*vi);
            let tang = vel_col - pd*dot(vel_col,pd);
            vel_col -= tang*0.5;
            av_y_col += cross(fwd, pd).y * vi * 0.05;
            pos_col  -= pd * 0.5;
        }
    }
    // Probe 2
    if (voxel_at(p2.x, p2.z) == 2u) {
        let pd = nor0(p2 - c.pos);
        let vi = dot(vel_col, pd);
        if (vi > 0.0) {
            vel_col += pd * (-(1.0+0.05)*vi);
            let tang = vel_col - pd*dot(vel_col,pd);
            vel_col -= tang*0.5;
            av_y_col += cross(fwd, pd).y * vi * 0.05;
            pos_col  -= pd * 0.5;
        }
    }
    // Probe 3
    if (voxel_at(p3.x, p3.z) == 2u) {
        let pd = nor0(p3 - c.pos);
        let vi = dot(vel_col, pd);
        if (vi > 0.0) {
            vel_col += pd * (-(1.0+0.05)*vi);
            let tang = vel_col - pd*dot(vel_col,pd);
            vel_col -= tang*0.5;
            av_y_col += cross(fwd, pd).y * vi * 0.05;
            pos_col  -= pd * 0.5;
        }
    }

    // Pre-integration ground clamp
    let gnd_c = height_at(c.pos.x, c.pos.z) + 0.5;
    if (pos_col.y < gnd_c) {
        pos_col.y = gnd_c;
        vel_col.y = max(vel_col.y, 0.0);
    }

    // === PHASE 5: semi-implicit Euler integration ===
    // NOTE: aerodynamic drag is NOT reapplied here.
    // Kernel A (physics_a.wgsl) already added drag to m.forces, which arrived in m2.forces.
    // Reapplying it caused 2× drag → speed ceiling 12 km/h lower (measured 237 vs 248 km/h).
    // The "drag_int" line was removed on 2026-08-03 after a confirmed bisection.
    let speed_int = length(vel_col);
    // drag_int removed — drag is already in m2.forces (from Kernel A)

    let accel     = m2.forces / c.mass;
    var final_vel = vel_col + accel * dt_safe;

    // Lateral scrub friction (CPU lines 727-735)
    let fwd_spd = dot(final_vel, fwd);
    let lat_v   = final_vel - fwd*fwd_spd;
    if (length(lat_v) > 0.5) {
        let scrub = max(1.0-3.0*dt_safe, 0.85);
        final_vel = fwd*fwd_spd + lat_v*scrub;
    }

    var final_pos = pos_col + final_vel*dt_safe;
    if (final_pos.y < 0.5) { final_pos.y=0.5; final_vel.y=max(final_vel.y,0.0); }

    // Yaw damping (speed-scalable — CPU v10)
    let drag_y    = -av_y_col * (400.0 + speed_int*60.0);
    let yi        = select(c.mass*2.0, m.yaw_inertia, m.yaw_inertia>0.0);
    let final_av_y= av_y_col + (m2.torques_y + drag_y)/yi * dt_safe;

    // === Writes the final output ===
    var out: CarOutput;
    out.vel          = final_vel;
    out.av_y         = final_av_y;
    out.pos          = final_pos;
    out._pad_pos     = 0.0;
    out.fwd          = fwd;
    out._pad_fwd     = 0.0;
    out.lat          = c.lat;
    out._pad_lat     = 0.0;
    out.current_rpm  = m.current_rpm;
    out.drive_torque_out = m.drive_torque_out;
    out.shift_timer  = m.shift_timer;
    out.is_stalled   = m.is_stalled;
    out.starter_timer= m.starter_timer;
    out.engine_temp  = m.engine_temp;
    out.oil_temp     = m.oil_temp;
    out._pad_eng     = 0.0;

    // Wheels: omegas and slip from Kernel B, susp_comps from Kernel A, tire_temps from the input
    let sc = m.susp_comps;
    let pc = m.prev_comps;
    let nl = m.normal_loads;

    // FL (i=0)
    out.wheels[0] = vec4<f32>(m2.omegas.x, m2.slip_r.x, m2.slip_a.x, nl.x);
    out.wheels[1] = vec4<f32>(c.wheels[1].x, sc.x, pc.x, 0.0);  // tire_temp from the input (thermal model in Kernel B not implemented to save registers)
    // FR (i=1)
    out.wheels[2] = vec4<f32>(m2.omegas.y, m2.slip_r.y, m2.slip_a.y, nl.y);
    out.wheels[3] = vec4<f32>(c.wheels[3].x, sc.y, pc.y, 0.0);
    // RL (i=2)
    out.wheels[4] = vec4<f32>(m2.omegas.z, m2.slip_r.z, m2.slip_a.z, nl.z);
    out.wheels[5] = vec4<f32>(c.wheels[5].x, sc.z, pc.z, 0.0);
    // RR (i=3)
    out.wheels[6] = vec4<f32>(m2.omegas.w, m2.slip_r.w, m2.slip_a.w, nl.w);
    out.wheels[7] = vec4<f32>(c.wheels[7].x, sc.w, pc.w, 0.0);

    output[being] = out;
}
