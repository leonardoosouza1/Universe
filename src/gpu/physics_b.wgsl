// Kernel B: loop de pneus (Pacejka) — deliberadamente enxuto para caber nos registradores.
//
// Causa raiz do crash original: o shader physics_b.wgsl combinava pneus + colisão + integração
// num único kernel com ~30 variáveis locais por iteração Pacejka somadas a 5 arrays de 4
// floats cada (om, tire_temps, slip_ratios, slip_angles, nl) — total ~70 f32 + vec3 forces
// + torques_y + demais. O compilador ACO/NIR da RDNA2 não consegue spill-ear para memória
// local e crasha com SIGSEGV.
//
// Solução: extrair SOMENTE o loop Pacejka para este kernel. A saída (MidState2, 64 bytes)
// carrega forças + torques + omegas atualizadas + slip para o Kernel C completar.

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

// Buffer intermediário do Kernel A — mesmo layout de physics_a.wgsl
struct MidState {
    normal_loads:    vec4<f32>,  // 16
    susp_comps:      vec4<f32>,  // 16
    prev_comps:      vec4<f32>,  // 16
    current_rpm:     f32,        //  4
    drive_torque:    f32,        //  4
    drive_torque_out:f32,        //  4
    shift_timer:     f32,        //  4
    is_stalled:      u32,        //  4
    starter_timer:   f32,        //  4
    engine_temp:     f32,        //  4
    oil_temp:        f32,        //  4
    omegas:          vec4<f32>,  // 16
    forces:          vec3<f32>,  // 12
    _pad_f:          f32,        //  4
    torques_y:       f32,        //  4
    yaw_inertia:     f32,        //  4
    _pad2:           vec2<f32>,  //  8
    // Total: 128 bytes
}

// Buffer intermediário do Kernel B — produzido aqui, consumido pelo Kernel C.
// Mantém MÍNIMO necessário para a integração. 64 bytes, stride=64.
struct MidState2 {
    forces:      vec3<f32>,   // 12  — forças de pneu + forças parciais do Kernel A
    torques_y:   f32,         //  4  — torque yaw acumulado
    omegas:      vec4<f32>,   // 16  — omegas atualizados pós-frenagem
    slip_r:      vec4<f32>,   // 16  — slip ratios (para atualizar wheels na saída)
    slip_a:      vec4<f32>,   // 16  — slip angles (idem)
    // Total: 64 bytes
}

struct WorldParams {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    _pad:      f32,
    // D5 (escada): clima — ver physics.rs::set_clima. Defaults neutros.
    clima_mu:   f32,
    vento_x:    f32,
    vento_z:    f32,
    clima_freq: f32,
}

@group(0) @binding(0) var<uniform>             params: WorldParams;
@group(0) @binding(1) var<storage, read>       grid:   array<u32>;  // não usado neste kernel, mas mantido para layout uniforme
@group(0) @binding(2) var<storage, read>       hmap:   array<f32>;  // idem
@group(0) @binding(3) var<storage, read>       input:  array<CarInput>;
@group(0) @binding(4) var<storage, read>       mid:    array<MidState>;
@group(0) @binding(5) var<storage, read_write> mid2:   array<MidState2>;

fn flag(flags: u32, bit: u32) -> bool { return (flags>>bit&1u)!=0u; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let being = gid.x;
    if (being >= params.n_beings) { return; }

    let c = input[being];
    let m = mid[being];
    let dt_safe = max(c.dt, 0.001);

    let speed   = length(c.vel);
    let fwd     = c.fwd;
    let hwb     = c.wheel_base * 0.5;
    let htw     = c.track_width * 0.5;
    let fwd_vel = dot(c.vel, fwd);
    let lat_vel = dot(c.vel, c.lat);
    // D5: μ local por posição (trechos escorregadios determinísticos). freq=0 → 1,0.
    var mu_local: f32 = 1.0;
    if (params.clima_freq > 0.0) {
        mu_local = params.clima_mu + (1.0 - params.clima_mu) * 0.5
                   * (1.0 + sin(c.pos.x * params.clima_freq) * cos(c.pos.z * params.clima_freq));
    } else if (params.clima_mu < 1.0) {
        mu_local = params.clima_mu;
    }
    let mu_base = c.mu_peak * c.terrain_grip * c.env_grip * mu_local;
    let lsd_gate= clamp(speed / 5.0, 0.0, 1.0);
    let dtq     = m.drive_torque;
    let aqua    = c.aquaplaning;

    // Carrega forças parciais do Kernel A (gravidade + suspensão + aero)
    var forces    = m.forces;
    var torques_y = m.torques_y;

    // Posições das rodas
    let wlp = array<f32,4>(hwb, hwb, -hwb, -hwb);
    let wtp = array<f32,4>(-htw, htw, -htw, htw);

    // Distribuição de torque — idêntico ao Kernel B antigo
    // Usa omegas de MidState (já atualizados pelo starter no Kernel A)
    var wt = array<f32,4>(0.0, 0.0, 0.0, 0.0);
    switch c.drivetrain {
        case 0u: { wt[0]=dtq*0.5; wt[1]=dtq*0.5; }
        case 1u: {
            let dw = abs(m.omegas.w - m.omegas.z);
            let lf = clamp(c.lsd_lock_ratio*dw*0.5, 0.0, 0.4);
            let sl = 0.5 + lf*sign(m.omegas.w - m.omegas.z);
            wt[2]=dtq*sl; wt[3]=dtq*(1.0-sl);
        }
        default: { wt[0]=dtq*0.25; wt[1]=dtq*0.25; wt[2]=dtq*0.25; wt[3]=dtq*0.25; }
    }

    let bt      = c.brake_cmd * c.brake_torque_max;
    let bpw     = array<f32,4>(bt*0.30, bt*0.30, bt*0.20, bt*0.20);

    var msd: f32;
    if (speed < 15.0) { msd = 35.0; }
    else              { msd = max(35.0 - (speed-15.0)*0.4, 12.0); }
    let msr = msd * 0.017453292519943295;

    // Omegas mutable (lidos do MidState, atualizados pelo loop)
    var om  = array<f32,4>(m.omegas.x, m.omegas.y, m.omegas.z, m.omegas.w);
    var sr4 = array<f32,4>(0.0, 0.0, 0.0, 0.0);  // slip ratios de saída
    var sa4 = array<f32,4>(0.0, 0.0, 0.0, 0.0);  // slip angles de saída

    // --- Loop Pacejka: 4 rodas ---
    // ATENÇÃO: variáveis por iteração deliberadamente mínimas para reduzir pressão
    // de registradores. Cada variável usada uma vez, sem retenção entre iterações.

    // Roda 0 (FL)
    {
        let i = 0u;
        let load = max(m.normal_loads.x, 10.0);
        var tf: f32 = 1.0;
        if (flag(c.config_flags,1u)) { tf = clamp(1.0-abs(c.wheels[1].x-c.optimal_temp)*0.5, 0.7, 1.0); }
        let mu  = mu_base * tf;
        let wls = om[i] * c.wheel_radius;
        let rfwd = fwd_vel + c.av_y * wtp[i];
        let rlat = lat_vel - c.av_y * wlp[i];
        let ws  = c.steer * msr;
        let cs  = cos(ws); let ss = sin(ws);
        let cf  = rfwd*cs + rlat*ss;
        let cl  = -rfwd*ss + rlat*cs;
        let vr  = max(abs(wls), max(abs(cf), 0.5));
        let slr = clamp((wls-cf)/vr, -1.0, 1.0);
        let fl  = load * mu * sin(1.5 * atan(10.0*slr));
        var sa: f32 = 0.0; var fla: f32;
        if (abs(cf) > 0.5) { sa = atan2(cl, abs(cf)); fla = load*mu*sin(1.35*atan(12.0*sa)); }
        else                { fla = clamp(cl*load*mu*2.0, -load*mu, load*mu); }
        let mf = mu*load; let comb = length(vec2<f32>(fl, fla));
        var fll: f32; var flal: f32;
        if (comb>mf && comb>0.001) { let sc=mf/comb; fll=fl*sc; flal=fla*sc; }
        else { fll=fl; flal=fla; }
        var no = om[i] + (wt[i]-fll*c.wheel_radius)/c.wheel_inertia * dt_safe;
        let brk = bpw[i];
        if (brk > 0.0) {
            let st=abs(no)*c.wheel_inertia/dt_safe; var ab: f32;
            if (flag(c.config_flags,5u)){ab=min(brk,st);}else{ab=brk;}
            if (no>0.0){no-=(ab/c.wheel_inertia)*dt_safe;}else if (no<0.0){no+=(ab/c.wheel_inertia)*dt_safe;}
        }
        om[i] = no;
        sr4[i] = slr; sa4[i] = sa;
        let chl  = fll*cs - flal*ss;
        let chla = fll*ss + flal*cs;
        forces    += fwd*chl*aqua - c.lat*chla*aqua;
        torques_y += chla*wlp[i]*aqua + chl*wtp[i]*0.005*lsd_gate;
    }

    // Roda 1 (FR)
    {
        let i = 1u;
        let load = max(m.normal_loads.y, 10.0);
        var tf: f32 = 1.0;
        if (flag(c.config_flags,1u)) { tf = clamp(1.0-abs(c.wheels[3].x-c.optimal_temp)*0.5, 0.7, 1.0); }
        let mu  = mu_base * tf;
        let wls = om[i] * c.wheel_radius;
        let rfwd = fwd_vel + c.av_y * wtp[i];
        let rlat = lat_vel - c.av_y * wlp[i];
        let ws  = c.steer * msr;
        let cs  = cos(ws); let ss = sin(ws);
        let cf  = rfwd*cs + rlat*ss;
        let cl  = -rfwd*ss + rlat*cs;
        let vr  = max(abs(wls), max(abs(cf), 0.5));
        let slr = clamp((wls-cf)/vr, -1.0, 1.0);
        let fl  = load * mu * sin(1.5 * atan(10.0*slr));
        var sa: f32 = 0.0; var fla: f32;
        if (abs(cf) > 0.5) { sa = atan2(cl, abs(cf)); fla = load*mu*sin(1.35*atan(12.0*sa)); }
        else                { fla = clamp(cl*load*mu*2.0, -load*mu, load*mu); }
        let mf = mu*load; let comb = length(vec2<f32>(fl, fla));
        var fll: f32; var flal: f32;
        if (comb>mf && comb>0.001) { let sc=mf/comb; fll=fl*sc; flal=fla*sc; }
        else { fll=fl; flal=fla; }
        var no = om[i] + (wt[i]-fll*c.wheel_radius)/c.wheel_inertia * dt_safe;
        let brk = bpw[i];
        if (brk > 0.0) {
            let st=abs(no)*c.wheel_inertia/dt_safe; var ab: f32;
            if (flag(c.config_flags,5u)){ab=min(brk,st);}else{ab=brk;}
            if (no>0.0){no-=(ab/c.wheel_inertia)*dt_safe;}else if (no<0.0){no+=(ab/c.wheel_inertia)*dt_safe;}
        }
        om[i] = no;
        sr4[i] = slr; sa4[i] = sa;
        let chl  = fll*cs - flal*ss;
        let chla = fll*ss + flal*cs;
        forces    += fwd*chl*aqua - c.lat*chla*aqua;
        torques_y += chla*wlp[i]*aqua + chl*wtp[i]*0.005*lsd_gate;
    }

    // Roda 2 (RL)
    {
        let i = 2u;
        let load = max(m.normal_loads.z, 10.0);
        var tf: f32 = 1.0;
        if (flag(c.config_flags,1u)) { tf = clamp(1.0-abs(c.wheels[5].x-c.optimal_temp)*0.5, 0.7, 1.0); }
        let mu  = mu_base * tf;
        let wls = om[i] * c.wheel_radius;
        let rfwd = fwd_vel + c.av_y * wtp[i];
        let rlat = lat_vel - c.av_y * wlp[i];
        // Rodas traseiras: sem esterçamento (select retorna 0.0 quando i >= 2)
        let cs  = 1.0; let ss = 0.0;
        let cf  = rfwd; let cl = rlat;
        let vr  = max(abs(wls), max(abs(cf), 0.5));
        let slr = clamp((wls-cf)/vr, -1.0, 1.0);
        let fl  = load * mu * sin(1.5 * atan(10.0*slr));
        var sa: f32 = 0.0; var fla: f32;
        if (abs(cf) > 0.5) { sa = atan2(cl, abs(cf)); fla = load*mu*sin(1.35*atan(12.0*sa)); }
        else                { fla = clamp(cl*load*mu*2.0, -load*mu, load*mu); }
        let mf = mu*load; let comb = length(vec2<f32>(fl, fla));
        var fll: f32; var flal: f32;
        if (comb>mf && comb>0.001) { let sc=mf/comb; fll=fl*sc; flal=fla*sc; }
        else { fll=fl; flal=fla; }
        var no = om[i] + (wt[i]-fll*c.wheel_radius)/c.wheel_inertia * dt_safe;
        let brk = bpw[i];
        if (brk > 0.0) {
            let st=abs(no)*c.wheel_inertia/dt_safe; var ab: f32;
            if (flag(c.config_flags,5u)){ab=min(brk,st);}else{ab=brk;}
            if (no>0.0){no-=(ab/c.wheel_inertia)*dt_safe;}else if (no<0.0){no+=(ab/c.wheel_inertia)*dt_safe;}
        }
        om[i] = no;
        sr4[i] = slr; sa4[i] = sa;
        // cs=1, ss=0 → chl=fll, chla=flal
        forces    += fwd*fll*aqua - c.lat*flal*aqua;
        torques_y += flal*wlp[i]*aqua + fll*wtp[i]*0.005*lsd_gate;
    }

    // Roda 3 (RR)
    {
        let i = 3u;
        let load = max(m.normal_loads.w, 10.0);
        var tf: f32 = 1.0;
        if (flag(c.config_flags,1u)) { tf = clamp(1.0-abs(c.wheels[7].x-c.optimal_temp)*0.5, 0.7, 1.0); }
        let mu  = mu_base * tf;
        let wls = om[i] * c.wheel_radius;
        let rfwd = fwd_vel + c.av_y * wtp[i];
        let rlat = lat_vel - c.av_y * wlp[i];
        // Rodas traseiras: sem esterçamento
        let cs  = 1.0; let ss = 0.0;
        let cf  = rfwd; let cl = rlat;
        let vr  = max(abs(wls), max(abs(cf), 0.5));
        let slr = clamp((wls-cf)/vr, -1.0, 1.0);
        let fl  = load * mu * sin(1.5 * atan(10.0*slr));
        var sa: f32 = 0.0; var fla: f32;
        if (abs(cf) > 0.5) { sa = atan2(cl, abs(cf)); fla = load*mu*sin(1.35*atan(12.0*sa)); }
        else                { fla = clamp(cl*load*mu*2.0, -load*mu, load*mu); }
        let mf = mu*load; let comb = length(vec2<f32>(fl, fla));
        var fll: f32; var flal: f32;
        if (comb>mf && comb>0.001) { let sc=mf/comb; fll=fl*sc; flal=fla*sc; }
        else { fll=fl; flal=fla; }
        var no = om[i] + (wt[i]-fll*c.wheel_radius)/c.wheel_inertia * dt_safe;
        let brk = bpw[i];
        if (brk > 0.0) {
            let st=abs(no)*c.wheel_inertia/dt_safe; var ab: f32;
            if (flag(c.config_flags,5u)){ab=min(brk,st);}else{ab=brk;}
            if (no>0.0){no-=(ab/c.wheel_inertia)*dt_safe;}else if (no<0.0){no+=(ab/c.wheel_inertia)*dt_safe;}
        }
        om[i] = no;
        sr4[i] = slr; sa4[i] = sa;
        // cs=1, ss=0 → chl=fll, chla=flal
        forces    += fwd*fll*aqua - c.lat*flal*aqua;
        torques_y += flal*wlp[i]*aqua + fll*wtp[i]*0.005*lsd_gate;
    }

    // Escreve MidState2 para o Kernel C
    var m2: MidState2;
    m2.forces   = forces;
    m2.torques_y= torques_y;
    m2.omegas   = vec4<f32>(om[0], om[1], om[2], om[3]);
    m2.slip_r   = vec4<f32>(sr4[0], sr4[1], sr4[2], sr4[3]);
    m2.slip_a   = vec4<f32>(sa4[0], sa4[1], sa4[2], sa4[3]);
    mid2[being] = m2;
}
