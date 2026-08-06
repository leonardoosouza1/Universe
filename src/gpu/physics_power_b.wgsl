// Bissecção FASE_POWER parte B: motor + loop de pneus SIMPLIFICADO (sem atan/Pacejka complexo).
// Inclui: arrays wheel_torques/brake_pw, loop de 4 rodas com cos/sin, mas force = 0.
// Objetivo: verificar se o crash é nos arrays adicionais ou no cálculo Pacejka.

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

const GRAVITY:     f32 = 9.81;
const PI2:         f32 = 6.283185307179586;
const FINAL_DRIVE: f32 = 3.7;

fn flag(flags: u32, bit: u32) -> bool { return (flags>>bit&1u)!=0u; }

fn height_at(wx: f32, wz: f32) -> f32 {
    let fx=wx/params.cell_size; let fz=wz/params.cell_size;
    let x0=i32(floor(fx)); let z0=i32(floor(fz));
    let tx=fx-f32(x0); let tz=fz-f32(z0);
    let n=i32(params.grid_size);
    var h00=0.0;var h10=0.0;var h01=0.0;var h11=0.0;
    if (x0>=0&&x0<n&&z0>=0&&z0<n){h00=hmap[u32(z0*n+x0)];}
    if ((x0+1)>=0&&(x0+1)<n&&z0>=0&&z0<n){h10=hmap[u32(z0*n+(x0+1))];}
    if (x0>=0&&x0<n&&(z0+1)>=0&&(z0+1)<n){h01=hmap[u32((z0+1)*n+x0)];}
    if ((x0+1)>=0&&(x0+1)<n&&(z0+1)>=0&&(z0+1)<n){h11=hmap[u32((z0+1)*n+(x0+1))];}
    return (h00*(1.0-tx)+h10*tx)*(1.0-tz)+(h01*(1.0-tx)+h11*tx)*tz;
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
    let speed = length(c.vel);

    let dpw = select(0.0,3.0*speed*speed,flag(c.config_flags,3u))*0.25;
    let sl0 = c.mass*GRAVITY/4.0;
    let fwd = c.fwd; let rgt = -c.lat;
    let hwb = c.wheel_base*0.5; let htw = c.track_width*0.5;
    var offs: array<vec3<f32>,4>;
    offs[0]=fwd*hwb-rgt*htw; offs[1]=fwd*hwb+rgt*htw;
    offs[2]=-fwd*hwb-rgt*htw; offs[3]=-fwd*hwb+rgt*htw;
    var nloads: array<f32,4>; var scomp: array<f32,4>;
    var tsl = 0.0;
    for (var i:u32=0u;i<4u;i++){
        let wp=c.pos+offs[i]; let gy=height_at(wp.x,wp.z);
        let comp=c.susp_rest-(wp.y-gy); let pc=c.wheels[i*2u+1u].z;
        if (comp>0.0&&comp<c.susp_max_travel){
            let tf=max(c.susp_stiffness*comp+c.susp_damping*(comp-pc)/dt_safe,0.0)+dpw;
            nloads[i]=clamp(tf,sl0*0.2,sl0*6.0); scomp[i]=comp;
        }else{nloads[i]=0.0;scomp[i]=0.0;}
        tsl+=nloads[i];
    }
    if (tsl<100.0){let pw=c.mass*GRAVITY*0.25;for (var i:u32=0u;i<4u;i++){nloads[i]=pw;}}

    let gratio=gear_ratio(c.current_gear);
    let total_ratio=gratio*FINAL_DRIVE; let tra=abs(total_ratio);
    var is_stalled=c.is_stalled; var starter_timer=c.starter_timer;
    var shift_timer=c.shift_timer; var current_rpm=c.current_rpm;
    var engine_temp=c.engine_temp; var oil_temp=c.oil_temp;
    var omegas: array<f32,4>; var tire_temps: array<f32,4>;
    var slip_ratios: array<f32,4>; var slip_angles: array<f32,4>;
    for (var i:u32=0u;i<4u;i++){
        omegas[i]=c.wheels[i*2u].x; tire_temps[i]=c.wheels[i*2u+1u].x;
        slip_ratios[i]=c.wheels[i*2u].y; slip_angles[i]=c.wheels[i*2u].z;
    }
    if (is_stalled!=0u&&flag(c.config_flags,0u)){
        if (c.current_gear<=2u&&c.throttle>0.5){
            starter_timer+=dt_safe;
            if (starter_timer>=1.0){is_stalled=0u;starter_timer=0.0;current_rpm=800.0;
                for (var i:u32=0u;i<4u;i++){omegas[i]=0.0;}}
        }else{starter_timer=0.0;}
    }
    if (shift_timer>0.0){shift_timer-=dt_safe;}
    var doa: f32;
    switch c.drivetrain {
        case 0u:{doa=(omegas[0]+omegas[1])*0.5;}
        case 1u:{doa=(omegas[2]+omegas[3])*0.5;}
        default:{doa=(omegas[0]+omegas[1]+omegas[2]+omegas[3])*0.25;}
    }
    let wrpm=abs(doa)*tra*60.0/PI2;
    let frpm=800.0+c.throttle*(c.max_rpm-800.0);
    var cb: f32;
    if (c.current_gear==1u){cb=0.0;}else if (wrpm<1500.0){cb=clamp(wrpm/1500.0,0.0,1.0);}else{cb=1.0;}
    var raw=frpm*(1.0-cb)+wrpm*cb;
    let cslip=cb<0.95;
    if (shift_timer>0.0){let fi=max(c.flywheel_inertia,0.05);raw=max(current_rpm-current_rpm*dt_safe/fi*50.0,800.0);}
    else if (flag(c.config_flags,0u)&&!cslip&&is_stalled==0u&&raw<550.0){is_stalled=1u;starter_timer=0.0;}
    if (is_stalled==0u&&flag(c.config_flags,1u)){
        let rr=current_rpm/c.max_rpm;
        engine_temp=clamp(engine_temp+rr*c.throttle*15.0*dt_safe-(speed*0.3+2.0)*dt_safe,20.0,150.0);
        oil_temp=oil_temp+(engine_temp-oil_temp)*0.01*dt_safe;
    }else{engine_temp=max(engine_temp-1.0*dt_safe,20.0);}
    var tp=1.0;
    if (flag(c.config_flags,1u)&&engine_temp>105.0){tp=clamp(1.0-(engine_temp-105.0)/45.0,0.1,1.0);}
    var dtq: f32; var dto: f32;
    if (is_stalled!=0u){current_rpm=0.0;dtq=0.0;dto=0.0;}
    else {
        current_rpm=clamp(raw,800.0,c.max_rpm+250.0);
        let rr=current_rpm/c.max_rpm;
        var tm=1.0-(rr-0.6)*(rr-0.6)*2.0;
        if (rr>1.0){tm=0.0;}else if (rr<0.2){tm*=rr/0.2;}
        tm=clamp(tm,0.0,1.0);
        var tcs=1.0;
        if (flag(c.config_flags,4u)&&c.drivetrain==1u){
            let sa=(max(slip_ratios[2],0.0)+max(slip_ratios[3],0.0))*0.5;
            if (sa>0.15){tcs=1.0-clamp((sa-0.15)*5.0,0.0,0.9);}
        }
        let ce=select(0.0,1.0,shift_timer<=0.0);
        dtq=c.max_torque*tm*c.throttle*tcs*total_ratio*ce*tp;
        dto=dtq;
    }

    // PARTE B: loop de pneus com distribuição de torque + frenos, MAS sem Pacejka/atan
    // (slip/force = 0, só integra omega)
    var wheel_torques = array<f32,4>(0.0,0.0,0.0,0.0);
    switch c.drivetrain {
        case 0u:{wheel_torques[0]=dtq*0.5;wheel_torques[1]=dtq*0.5;}
        case 1u:{
            let dw=abs(omegas[3]-omegas[2]);
            let lf=clamp(c.lsd_lock_ratio*dw*0.5,0.0,0.4);
            let sl=0.5+lf*sign(omegas[3]-omegas[2]);
            wheel_torques[2]=dtq*sl;wheel_torques[3]=dtq*(1.0-sl);
        }
        default:{for (var i:u32=0u;i<4u;i++){wheel_torques[i]=dtq*0.25;}}
    }
    let brake_total=c.brake_cmd*c.brake_torque_max;
    let brake_pw=array<f32,4>(brake_total*0.30,brake_total*0.30,brake_total*0.20,brake_total*0.20);
    var max_steer_deg: f32;
    if (speed<15.0){max_steer_deg=35.0;}else{max_steer_deg=max(35.0-(speed-15.0)*0.4,12.0);}
    let max_steer_rad=max_steer_deg*0.017453292519943295;

    // Loop de pneus: apenas cos/sin do esterço + integração de omega SEM cálculo de força
    for (var i:u32=0u;i<4u;i++){
        let ws=select(0.0,c.steer*max_steer_rad,i<2u);
        let cs=cos(ws);let ss=sin(ws);
        // Integração de omega sem force (force = 0 → só torque de tração)
        var no=omegas[i]+(wheel_torques[i]/c.wheel_inertia)*dt_safe;
        let brk=brake_pw[i];
        if (brk>0.0){
            let st=abs(no)*c.wheel_inertia/dt_safe;
            var ab: f32;
            if (flag(c.config_flags,5u)){ab=min(brk,st);}else{ab=brk;}
            if (no>0.0){no-=(ab/c.wheel_inertia)*dt_safe;}
            else if (no<0.0){no+=(ab/c.wheel_inertia)*dt_safe;}
        }
        omegas[i]=no;
        // Pequeno uso de cs/ss para que não sejam otimizados fora
        slip_ratios[i]=cs*0.001;
        slip_angles[i]=ss*0.001;
    }

    var out: CarOutput;
    out.vel=c.vel; out.av_y=c.av_y; out.pos=c.pos; out._pad_pos=0.0;
    out.fwd=fwd; out._pad_fwd=0.0; out.lat=c.lat; out._pad_lat=0.0;
    out.current_rpm=current_rpm; out.drive_torque_out=dto;
    out.shift_timer=max(shift_timer,0.0); out.is_stalled=is_stalled;
    out.starter_timer=starter_timer; out.engine_temp=engine_temp;
    out.oil_temp=oil_temp; out._pad_eng=0.0;
    for (var i:u32=0u;i<4u;i++){
        out.wheels[i*2u]=vec4<f32>(omegas[i],slip_ratios[i],slip_angles[i],nloads[i]);
        out.wheels[i*2u+1u]=vec4<f32>(tire_temps[i],scomp[i],0.0,0.0);
    }
    output[being]=out;
}
