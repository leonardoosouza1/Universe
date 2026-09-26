// Kernel A: suspension + engine (no tires, no collision, no integration).
// Produces an intermediate buffer that Kernel B will consume.
//
// Split motivated by register pressure on RADV NAVI22 (ACO): the full shader
// with Pacejka + all the local arrays SIGSEGVs the compiler.
// Kernel A uses ~80 local f32; Kernel B uses ~90 local f32. Both are below the threshold.

// --- original input ---
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

// --- intermediate buffer produced by Kernel A and consumed by Kernel B ---
// Contains: partial suspension result + engine state
struct MidState {
    // Normal load per wheel (N) — computed by the suspension
    normal_loads:    vec4<f32>,   // [FL, FR, RL, RR]
    susp_comps:      vec4<f32>,   // current spring compression [m]
    prev_comps:      vec4<f32>,   // compression at the previous tick (for damping)
    // Engine state after kernel A
    current_rpm:     f32,
    drive_torque:    f32,
    drive_torque_out:f32,
    shift_timer:     f32,
    is_stalled:      u32,
    starter_timer:   f32,
    engine_temp:     f32,
    oil_temp:        f32,
    // Wheel omegas (may be modified by the starter)
    omegas:          vec4<f32>,
    // Partial resulting force (gravity + suspension + aero + rolling + engine brake)
    // Kernel B adds the tire forces and completes the integration.
    forces:          vec3<f32>,
    _pad_f:          f32,
    // Yaw torque accumulator (tires will add more in Kernel B)
    torques_y:       f32,
    yaw_inertia:     f32,
    tcs_fraction: f32, // acquired controller gain, owned by kernel A
    tcs_initialized: u32,
}

struct WorldParams {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    _pad:      f32,
    // D5: climate mirror (see physics.rs). Kernel A uses only the wind.
    clima_mu:   f32,
    vento_x:    f32,
    vento_z:    f32,
    clima_freq: f32,
}

@group(0) @binding(0) var<uniform>             params: WorldParams;
@group(0) @binding(1) var<storage, read>       grid:   array<u32>;
@group(0) @binding(2) var<storage, read>       hmap:   array<f32>;
@group(0) @binding(3) var<storage, read>       input:  array<CarInput>;
@group(0) @binding(4) var<storage, read_write> mid:    array<MidState>;

const GRAVITY:      f32 = 9.81;
const RHO_AR:       f32 = 1.225;
const FRONTAL_AREA: f32 = 2.2;
const PI2:          f32 = 6.283185307179586;
const FINAL_DRIVE:  f32 = 3.7;

fn height_at(wx: f32, wz: f32) -> f32 {
    let fx = wx / params.cell_size;
    let fz = wz / params.cell_size;
    let x0 = i32(floor(fx)); let z0 = i32(floor(fz));
    let tx = fx - f32(x0); let tz = fz - f32(z0);
    let n  = i32(params.grid_size);
    var h00=0.0;var h10=0.0;var h01=0.0;var h11=0.0;
    if (x0>=0&&x0<n&&z0>=0&&z0<n)                 { h00=hmap[u32(z0*n+x0)]; }
    if ((x0+1)>=0&&(x0+1)<n&&z0>=0&&z0<n)         { h10=hmap[u32(z0*n+(x0+1))]; }
    if (x0>=0&&x0<n&&(z0+1)>=0&&(z0+1)<n)         { h01=hmap[u32((z0+1)*n+x0)]; }
    if ((x0+1)>=0&&(x0+1)<n&&(z0+1)>=0&&(z0+1)<n) { h11=hmap[u32((z0+1)*n+(x0+1))]; }
    return (h00*(1.0-tx)+h10*tx)*(1.0-tz)+(h01*(1.0-tx)+h11*tx)*tz;
}

fn flag(flags: u32, bit: u32) -> bool { return (flags>>bit&1u)!=0u; }

fn nor0(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l < 1e-9) { return vec3<f32>(0.0); }
    return v / l;
}

fn gear_ratio(g: u32) -> f32 {
    switch g {
        case 0u:{return -3.5;} case 1u:{return 0.0;} case 2u:{return 3.8;}
        case 3u:{return 2.6;} case 4u:{return 1.9;} case 5u:{return 1.45;}
        default:{return 1.15;}
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let being = gid.x;
    if (being >= params.n_beings) { return; }

    let c = input[being];
    let dt_safe = max(c.dt, 0.001);

    // === PHASE 1: reset forces ===
    var forces = vec3<f32>(0.0, -GRAVITY * c.mass, 0.0);
    // D5: constant lateral wind (N). Default 0 = unchanged.
    forces.x += params.vento_x;
    forces.z += params.vento_z;

    // === PHASE 2: suspension + normal load ===
    let speed = length(c.vel);
    let aero_down = select(0.0, 3.0*speed*speed, flag(c.config_flags,3u));
    let dpw = aero_down * 0.25;
    let static_load = c.mass * GRAVITY / 4.0;
    let fwd = c.fwd; let rgt = -c.lat;
    let hwb = c.wheel_base * 0.5; let htw = c.track_width * 0.5;

    var offs: array<vec3<f32>, 4>;
    offs[0] = fwd*hwb - rgt*htw;
    offs[1] = fwd*hwb + rgt*htw;
    offs[2] = -fwd*hwb - rgt*htw;
    offs[3] = -fwd*hwb + rgt*htw;

    var nl: array<f32,4>; var sc: array<f32,4>; var pc: array<f32,4>;
    var tsl = 0.0;
    for (var i:u32=0u;i<4u;i++) {
        let wp  = c.pos + offs[i];
        let gy  = height_at(wp.x, wp.z);
        let comp= c.susp_rest - (wp.y - gy);
        let prc = c.wheels[i*2u+1u].z;
        if (comp>0.0 && comp<c.susp_max_travel) {
            let tf = max(c.susp_stiffness*comp + c.susp_damping*(comp-prc)/dt_safe, 0.0) + dpw;
            nl[i] = clamp(tf, static_load*0.2, static_load*6.0);
            sc[i] = comp;
        } else { nl[i]=0.0; sc[i]=0.0; }
        pc[i] = comp;
        tsl  += nl[i];
        forces.y += nl[i];
    }
    if (tsl < 100.0) {
        let pw = c.mass*GRAVITY*0.25;
        for (var i:u32=0u;i<4u;i++) { nl[i]=pw; }
    }

    // === PHASE 3a: engine (no tire loop) ===
    let gratio       = gear_ratio(c.current_gear);
    let total_ratio  = gratio * FINAL_DRIVE;
    let tra          = abs(total_ratio);

    var is_stalled    = c.is_stalled;
    var starter_timer = c.starter_timer;
    var shift_timer   = c.shift_timer;
    var current_rpm   = c.current_rpm;
    var engine_temp   = c.engine_temp;
    var oil_temp      = c.oil_temp;

    let powertrain_v2 = flag(c.config_flags,13u);
    // Legacy restart resets wheels; v2 preserves their physical momentum.
    var om0=c.wheels[0].x; var om1=c.wheels[2].x;
    var om2=c.wheels[4].x; var om3=c.wheels[6].x;

    if (is_stalled!=0u && flag(c.config_flags,0u)) {
        let authorized = select(c.throttle>0.5, c._pad_wind>0.5 && speed<0.5 && c.shift_timer<=0.0, powertrain_v2);
        if (c.current_gear<=2u && authorized) {
            starter_timer += dt_safe;
            if (starter_timer>=1.0) {
                is_stalled=0u; starter_timer=0.0; current_rpm=800.0;
                if (!powertrain_v2) {om0=0.0; om1=0.0; om2=0.0; om3=0.0;}
            }
        } else { starter_timer=0.0; }
    }
    if (shift_timer>0.0) { shift_timer=max(0.0,shift_timer-dt_safe); }

    var doa: f32;
    switch c.drivetrain {
        case 0u: { doa=(om0+om1)*0.5; }
        case 1u: { doa=(om2+om3)*0.5; }
        default: { doa=(om0+om1+om2+om3)*0.25; }
    }
    let wrpm    = abs(doa)*tra*60.0/PI2;
    let frpm    = 800.0 + c.throttle*(c.max_rpm-800.0);
    var cb: f32;
    if (c.current_gear==1u)        { cb=0.0; }
    else if (wrpm<1500.0)          { cb=clamp(wrpm/1500.0,0.0,1.0); }
    else                           { cb=1.0; }
    // Bounded automatic launch clutch in first/reverse; higher forward gears
    // couple the engine to the wheels. This is not a calibrated clutch model.
    if (powertrain_v2 && c.current_gear>=3u) {cb=1.0;}
    var raw = frpm*(1.0-cb)+wrpm*cb;
    let cslip = cb < 0.95;
    if (shift_timer>0.0) {
        let fi=max(c.flywheel_inertia,0.05);
        raw=max(current_rpm - current_rpm*dt_safe/fi*50.0, 800.0);
    } else if (flag(c.config_flags,0u) && !cslip && is_stalled==0u && raw<550.0) {
        is_stalled=1u; starter_timer=0.0;
    }

    // No usable energy/capacity means engine-off, not a free idling engine.
    if (powertrain_v2 && c.max_torque<=0.0) {is_stalled=1u; starter_timer=0.0;}
    if (!flag(c.config_flags,8u)) {
    if (is_stalled==0u && flag(c.config_flags,1u)) {
        let rr=current_rpm/c.max_rpm;
        var heat = rr * c.throttle * 15.0 * dt_safe;
        var cooling = (speed * 0.3 + 2.0) * dt_safe;
        if flag(c.config_flags,7u) {
            heat = heat * (c.max_torque / 620.0);
            cooling = cooling * (1.0 + c._pad_before_wind);
        }
        engine_temp=clamp(engine_temp+heat-cooling,20.0,150.0);
        oil_temp=oil_temp+(engine_temp-oil_temp)*0.01*dt_safe;
    } else { engine_temp=max(engine_temp-1.0*dt_safe,20.0); }

    } // organism-taxi-v1 owns thermal integration when bit 8 is set.

    var tp=1.0;
    if (flag(c.config_flags,1u) && engine_temp>105.0) {
        tp=clamp(1.0-(engine_temp-105.0)/45.0,0.1,1.0);
    }

    var dtq: f32; var dto: f32;
    var stored_tcs=1.0;
    if (is_stalled!=0u) {
        current_rpm=0.0; dtq=0.0; dto=0.0;
    } else {
        current_rpm=clamp(raw,800.0,c.max_rpm+250.0);
        let rr=current_rpm/c.max_rpm;
        var tm=1.0-(rr-0.6)*(rr-0.6)*2.0;
        if (rr>1.0){tm=0.0;} else if (rr<0.2){tm*=rr/0.2;}
        tm=clamp(tm,0.0,1.0);
        var tcs=1.0;
        // TCS uses slip_ratios[2] and [3] — reads directly from c.wheels (not from local arrays)
        if (flag(c.config_flags,4u) && c.drivetrain==1u) {
            let sa=(max(c.wheels[4].y,0.0)+max(c.wheels[6].y,0.0))*0.5;
            if (sa>0.15) { tcs=1.0-clamp((sa-0.15)*5.0,0.0,0.9); }
        }
        if (flag(c.config_flags,12u) && flag(c.config_flags,4u) && c.drivetrain==1u) {
            // Filter only the traction-control gain, never the pedal request.
            // Zero-initialized/reset state means no prior traction intervention.
            let previous=select(1.0,clamp(mid[being].tcs_fraction,0.1,1.0),mid[being].tcs_initialized==1u);
            let recovery=1.0-exp(-dt_safe/0.1);
            tcs=min(tcs,previous+(tcs-previous)*recovery);
        }
        stored_tcs=tcs;
        let ce=select(0.0,1.0,shift_timer<=0.0);
        dtq=c.max_torque*tm*c.throttle*tcs*total_ratio*ce*tp;
        dto=dtq;
    }

    // === PHASE 3b: aero + rolling + engine brake (no tires) ===
    forces += c.wind * c.mass * 0.5;
    if (flag(c.config_flags,2u)) {
        let drag_mag=0.5*RHO_AR*speed*speed*c.drag_coeff*FRONTAL_AREA;
        forces += -nor0(c.vel)*drag_mag;
    }
    if (flag(c.config_flags,3u)) {
        forces.y -= 0.5*RHO_AR*speed*speed*c.drag_coeff*2.5*FRONTAL_AREA;
    }
    let total_load_rr = nl[0]+nl[1]+nl[2]+nl[3];
    if (speed>0.05) { forces += -nor0(c.vel)*(0.015*total_load_rr); }

    let fwd_vel = dot(c.vel, fwd);
    if (is_stalled==0u && c.throttle<0.05 && abs(fwd_vel)>1.0) {
        let eb  = tra*80.0;
        let meb = abs(fwd_vel)*c.mass/dt_safe;
        forces += -fwd*sign(fwd_vel)*min(eb,meb);
    }

    let yaw_inertia = c.mass*(hwb*hwb);

    // === Writes the intermediate buffer ===
    var m: MidState;
    m.normal_loads = vec4<f32>(nl[0],nl[1],nl[2],nl[3]);
    m.susp_comps   = vec4<f32>(sc[0],sc[1],sc[2],sc[3]);
    m.prev_comps   = vec4<f32>(pc[0],pc[1],pc[2],pc[3]);
    m.current_rpm  = current_rpm;
    m.drive_torque = dtq;
    m.drive_torque_out = dto;
    m.shift_timer  = max(shift_timer, 0.0);
    m.is_stalled   = is_stalled;
    m.starter_timer= starter_timer;
    m.engine_temp  = engine_temp;
    m.oil_temp     = oil_temp;
    m.omegas       = vec4<f32>(om0,om1,om2,om3);
    m.forces       = forces;
    m._pad_f       = 0.0;
    m.torques_y    = 0.0;  // tires will add in Kernel B
    m.yaw_inertia  = yaw_inertia;
    m.tcs_fraction = stored_tcs;
    m.tcs_initialized = 1u;
    mid[being] = m;
}
