// FÍSICA VEICULAR NA GPU — um thread por carro, kernel único por passo de simulação.
//
// # Decisão de projeto: kernel único vs quatro kernels separados
//
// A CPU roda suspensão → powertrain → colisão → integração em sequência, e CADA FASE lê
// resultado da fase anterior DO MESMO CARRO. Isso descarta pipelines assíncronos porque a
// dependência é local (carro i depende só de si mesmo). Com um único kernel:
//
//   1. Sem barreiras de memória global entre fases — cada thread carrega o estado do seu carro
//      em registradores, executa as 4 fases em sequência e escreve de volta uma única vez.
//   2. Sem overhead de 4 dispatches + 4 bind groups + 4 syncs.
//   3. A GPU tem ~1800 registradores f32 por workgroup RDNA2; um carro usa ~150 floats.
//      Não medido — estimativa conservadora. Pressão de registrador não deve ser gargalo.
//
// Com 60 seres cabe em 1 workgroup de 64 threads (threads 60-63 saem no guard inicial).
// O ponto de saturação de compute só aparece acima de ~512 threads; em 60 o gargalo é
// latência de round-trip host↔GPU, não ALU. O ganho principal vem do estado residente.
//
// # Riscos de divergência aritmética vs CPU
//
// 1. `atan` e `sin` WGSL: IEEE-754 single-precision, fiel a ±1 ULP na maioria das
//    implementações Vulkan/GL. A GPU pode usar instrução nativa que difere da libm do x86.
//    Ponto sensível: Pacejka (sin(c * atan(b * sr))) e slip_angle (atan2). Documentado
//    in-situ.
//
// 2. `sqrt` / `length`: ±1 ULP vs sqrtf da CPU. Crítico no círculo de Kamm.
//
// 3. Ordem de acumulação de forças: igual à CPU (acumula em variáveis locais, aplica ao final).
//
// 4. Interpolação bilinear do heightmap: floor() GPU vs CPU — ambos seguem IEEE 60559. OK.
//
// 5. SUPOSIÇÃO MORTA em physics.rs linha 631:
//    `yaw_inertia = mass * (wheel_base/2)^2`
//    É o momento de inércia de um ponto de massa; ignora track_width e distribuição real.
//    O comentário "momento de inércia correto" é otimista. Replicamos a mesma fórmula.
//
// 6. POSSÍVEL BUG na CPU physics.rs linha 467:
//    `split_left = 0.5 + lock_factor * signum(omega_R - omega_L)`
//    split_left não é clampeado a [0, 1] — pode ser >1.0 com lock_factor alto.
//    split_right = 1.0 - split_left pode ser negativo. Replicamos fielmente.
//
// 7. DUPLICAÇÃO DE AERO DRAG na CPU: powertrain_and_tire_system aplica drag (fase E) E
//    integration_system reaplica (linha 717-721). Replicamos os dois.
//
// 8. YAW KICK DE COLISÃO: a CPU (linha 687) adiciona diretamente a body.angular_velocity.y
//    — não via torque. Aqui usamos av_y_col acumulado, adicionado ANTES da integração.

// ============================================================
// === STRUCTS DE ENTRADA/SAÍDA                             ===
// ============================================================

// Estado completo de um carro para leitura. Espelho de EstadoFisicoGpu em fisica.rs.
// Todos os vec3 têm padding de f32 para alinhamento de 16 bytes exigido pelo WGSL.
struct CarInput {
    // --- PhysicsBody ---
    vel:        vec3<f32>,
    av_y:       f32,         // angular_velocity.y (rad/s)
    // 4 rodas × 2 vec4 = 8 vec4. Roda i: wheels[i*2] = [omega, slip_r, slip_a, normal_load]
    //                                    wheels[i*2+1] = [tire_temp, susp_comp, prev_comp, pad]
    wheels:     array<vec4<f32>, 8>,
    // --- Transform ---
    pos:        vec3<f32>,
    _pad_pos:   f32,
    fwd:        vec3<f32>,   // transform.right() em Bevy = "frente" no physics.rs
    _pad_fwd:   f32,
    lat:        vec3<f32>,   // transform.back() em Bevy = "lateral" no physics.rs
    _pad_lat:   f32,
    // --- EngineComponent ---
    max_torque:      f32,
    current_rpm:     f32,
    max_rpm:         f32,
    throttle:        f32,
    brake_cmd:       f32,
    steer:           f32,
    drive_torque_out:f32,
    current_gear:    u32,   // 0=ré, 1=neutro, 2=1ª … 6=6ª
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
    // PADDING: offset após susp_max_travel = 316 (não múltiplo de 16).
    // wind: vec3<f32> exige alinhamento 16 → inserir 4 bytes aqui para levar ao offset 320.
    _pad_before_wind: f32,
    // --- Ambiente (snapshot do tick) ---
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
// === CONSTANTES                                           ===
// ============================================================

const GRAVITY:      f32 = 9.81;
const RHO_AR:       f32 = 1.225;   // kg/m³ densidade do ar ao nível do mar
const FRONTAL_AREA: f32 = 2.2;     // m² area frontal GT3 (não medido, assumido no physics.rs)
const PI2:          f32 = 6.283185307179586;

// ============================================================
// === AUXILIARES DE MUNDO                                  ===
// ============================================================

// Idêntico a get_voxel_at em world.rs: round() + índice [z*size+x].
// Retorna Wall (2) fora dos limites — igual à CPU.
fn voxel_at(wx: f32, wz: f32) -> u32 {
    let x = i32(round(wx / params.cell_size));
    let z = i32(round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if (x >= 0 && x < n && z >= 0 && z < n) {
        return grid[u32(z * n + x)];
    }
    return 2u; // Wall
}

// Idêntico a get_height_at em world.rs: interpolação bilinear.
// DIVERGÊNCIA POSSÍVEL: floor() é especificado IEEE 60559 em ambos — OK para inputs normais.
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

// normalize_or_zero: igual a Bevy Vec3::normalize_or_zero.
// Separado do builtin `normalize` para evitar NaN quando v = 0.
fn nor0(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l < 1e-9) { return vec3<f32>(0.0, 0.0, 0.0); }
    return v / l;
}

// Extrai bit `bit` de config_flags. Recebe o valor diretamente (não ponteiro para struct)
// para evitar dependência do address space de EntradaCarro.
fn flag(flags: u32, bit: u32) -> bool {
    return (flags >> bit & 1u) != 0u;
}

// Razão de transmissão por marcha. Índice: 0=ré, 1=neutro, 2=1ª..6=6ª.
// Valores idênticos ao array `gears` da CPU (physics.rs linha 263).
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
// === KERNEL PRINCIPAL                                     ===
// ============================================================

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let being = gid.x;
    if (being >= params.n_beings) { return; }

    // Cópia local da entrada (leitura única do storage buffer — depois operam os registradores)
    var c = input[being];

    let dt_safe = max(c.dt, 0.001);  // mesmo guard da CPU: `time.delta_secs().max(0.001)`

    // ============================================================
    // === FASE 1: RESET FORCES                                 ===
    // ============================================================
    var forces = vec3<f32>(0.0, -GRAVITY * c.mass, 0.0);
    var torques_y = 0.0;  // só yaw importa no modelo 2.5D

    // ============================================================
    // === FASE 2: SUSPENSÃO + CARGA NORMAL                    ===
    // ============================================================
    let speed = length(c.vel);
    let aero_downforce = select(0.0, 3.0 * speed * speed, flag(c.config_flags,3u));
    let downforce_per_wheel = aero_downforce * 0.25;
    let static_load = c.mass * GRAVITY / 4.0;

    let fwd = c.fwd;
    let rgt = -c.lat;   // right real = -back() do Bevy

    let half_wb = c.wheel_base * 0.5;
    let half_tw = c.track_width * 0.5;

    // Offsets de roda [FL, FR, RL, RR] — igual à CPU (suspension_and_load_system)
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

    // Fallback: sem carga em nenhuma roda → peso estático por roda
    let total_weight = c.mass * GRAVITY;
    if (total_susp_load < 100.0) {
        for (var i: u32 = 0u; i < 4u; i++) {
            normal_loads[i] = total_weight * 0.25;
        }
    }

    // ============================================================
    // === FASE 3: POWERTRAIN + PNEUS + AERO                   ===
    // ============================================================
    let forward_dir = fwd;
    let lateral_dir = c.lat;    // back() do Bevy — igual à CPU
    let forward_vel = dot(c.vel, forward_dir);
    let lateral_vel = dot(c.vel, lateral_dir);

    let gratio        = gear_ratio(c.current_gear);
    let total_ratio   = gratio * FINAL_DRIVE;
    let total_ratio_abs = abs(total_ratio);

    // Estado mutável do motor (copiado para var locais)
    var is_stalled      = c.is_stalled;
    var starter_timer   = c.starter_timer;
    var shift_timer     = c.shift_timer;
    var current_rpm     = c.current_rpm;
    var engine_temp     = c.engine_temp;
    var oil_temp        = c.oil_temp;

    // Estado mutável das rodas
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

    // --- A) Motor de arranque ---
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

    // --- B) Transmissão ---
    // Troca de marcha (manual_shift_req) gerenciada na CPU antes do dispatch — o host
    // entrega current_gear já atualizado. Aqui só decaímos o shift_timer.
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

    // --- D) Térmica do motor ---
    if (is_stalled == 0u && flag(c.config_flags,1u)) {
        let rpm_ratio = current_rpm / c.max_rpm;
        let heat_gen  = rpm_ratio * c.throttle * 15.0 * dt_safe;
        let cooling   = (speed * 0.3 + 2.0) * dt_safe;
        engine_temp = clamp(engine_temp + heat_gen - cooling, 20.0, 150.0);
        oil_temp    = oil_temp + (engine_temp - oil_temp) * 0.01 * dt_safe;
    } else {
        engine_temp = max(engine_temp - 1.0 * dt_safe, 20.0);
    }

    var thermal_power = 1.0;
    if (flag(c.config_flags,1u) && engine_temp > 105.0) {
        thermal_power = clamp(1.0 - (engine_temp - 105.0) / 45.0, 0.1, 1.0);
    }

    // --- Torque de acionamento ---
    var drive_torque:     f32;
    var drive_torque_out: f32;
    if (is_stalled != 0u) {
        current_rpm      = 0.0;
        drive_torque     = 0.0;
        drive_torque_out = 0.0;
    } else {
        current_rpm = clamp(raw_rpm, 800.0, c.max_rpm + 250.0);
        let rpm_ratio = current_rpm / c.max_rpm;

        // Curva de torque realística (pico em ~60% RPM) — physics.rs linha 382
        var tm = 1.0 - (rpm_ratio - 0.6) * (rpm_ratio - 0.6) * 2.0;
        if (rpm_ratio > 1.0) {
            tm = 0.0;
        } else if (rpm_ratio < 0.2) {
            tm *= rpm_ratio / 0.2;
        }
        tm = clamp(tm, 0.0, 1.0);

        // TCS — só RWD
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

    // --- E) Aerodinâmica ---
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

    // Engine braking (igual CPU linha 432-438)
    if (is_stalled == 0u && c.throttle < 0.05 && abs(forward_vel) > 1.0) {
        let eb = total_ratio_abs * 80.0;
        let max_eb = abs(forward_vel) * c.mass / dt_safe;
        forces += -forward_dir * sign(forward_vel) * min(eb, max_eb);
    }

    // --- F) Grip de terreno e ambiente ---
    let mu_base = c.mu_peak * c.terrain_grip * c.env_grip;

    // --- G) Pneus por roda ---
    let wheel_long_pos = array<f32, 4>(half_wb, half_wb, -half_wb, -half_wb);
    let wheel_lat_pos  = array<f32, 4>(-half_tw, half_tw, -half_tw, half_tw);

    // LSD / distribuição de torque
    var wheel_torques = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    switch c.drivetrain {
        case 0u: {  // FWD — split uniforme
            wheel_torques[0] = drive_torque * 0.5;
            wheel_torques[1] = drive_torque * 0.5;
        }
        case 1u: {  // RWD — LSD
            let delta_omega  = abs(omegas[3] - omegas[2]);
            let lock_factor  = clamp(c.lsd_lock_ratio * delta_omega * 0.5, 0.0, 0.4);
            // POSSÍVEL BUG CPU replicado: split_left não clampeado a [0,1]
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

    // Bias de freio: 30/30/20/20
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

    // Speed-sensitive steering (CPU linha 518-524)
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

        // Temperatura do pneu
        var temp_factor = 1.0;
        if (flag(c.config_flags,1u)) {
            temp_factor = clamp(1.0 - abs(tire_temps[i] - c.optimal_temp) * 0.5, 0.7, 1.0);
        }
        let mu = mu_base * temp_factor;

        let wheel_linear_speed = omegas[i] * c.wheel_radius;

        // Velocidade no ponto de contato (inclui yaw angular — physics.rs linhas 514-515)
        let raw_fwd = forward_vel + c.av_y * wheel_lat_pos[i];
        let raw_lat = lateral_vel - c.av_y * wheel_long_pos[i];

        // Ângulo de esterçamento (só rodas dianteiras: i < 2)
        let wheel_steer = select(0.0, c.steer * max_steer_rad, i < 2u);
        let cos_s = cos(wheel_steer);
        let sin_s = sin(wheel_steer);

        // Velocidade no referencial da roda
        let contact_fwd = raw_fwd * cos_s + raw_lat * sin_s;
        let contact_lat = -raw_fwd * sin_s + raw_lat * cos_s;

        // Slip ratio (Pacejka longitudinal)
        // DIVERGÊNCIA POSSÍVEL: max() e clamp() são determinísticos; divisão IEEE-754. ±1 ULP.
        let v_ref = max(abs(wheel_linear_speed), max(abs(contact_fwd), 0.5));
        let sr    = clamp((wheel_linear_speed - contact_fwd) / v_ref, -1.0, 1.0);

        // Pacejka longitudinal: F = load * mu * sin(c_long * atan(b_long * sr))
        // DIVERGÊNCIA POSSÍVEL: atan GPU pode diferir de atanf CPU em ±1 ULP.
        let b_long: f32 = 10.0;
        let c_long: f32 = 1.5;
        let f_long = load * mu * sin(c_long * atan(b_long * sr));

        // Pacejka lateral
        var sa:    f32 = 0.0;
        var f_lat: f32;
        if (abs(contact_fwd) > 0.5) {
            // DIVERGÊNCIA POSSÍVEL: atan2 GPU vs atan2f CPU ±1 ULP.
            sa    = atan2(contact_lat, abs(contact_fwd));
            let b_lat: f32 = 12.0;
            let c_lat: f32 = 1.35;
            f_lat = load * mu * sin(c_lat * atan(b_lat * sa));
        } else {
            // Bug gelo fix: grip sintético viscoso (CPU linha 553)
            f_lat = clamp(contact_lat * load * mu * 2.0, -load * mu, load * mu);
        }

        // Círculo de Kamm
        // DIVERGÊNCIA POSSÍVEL: length() GPU vs hypotf CPU ±1 ULP.
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

        // Integração da omega da roda
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

        // Aquaplanagem (flag booleana passada como 0.1 ou 1.0 pelo host)
        let aqua = c.aquaplaning;

        // Rotação de volta para o referencial do chassi
        let ch_long = f_long_f * cos_s - f_lat_f * sin_s;
        let ch_lat  = f_long_f * sin_s + f_lat_f * cos_s;

        total_long_force += ch_long * aqua;
        total_lat_force  += ch_lat  * aqua;
        total_yaw_torque += ch_lat  * wheel_long_pos[i] * aqua;
        total_yaw_torque += ch_long * wheel_lat_pos[i]  * 0.005 * lsd_speed_gate;

        // Térmica do pneu
        if (flag(c.config_flags,1u)) {
            let slip_heat  = (abs(sr) + abs(sa)) * load * 0.000002 * dt_safe;
            // NOTA: brake_heat usa new_omega (pós-frenagem), igual à CPU que usa o valor
            // atualizado na mesma iteração (writeback ocorre no final do loop na CPU).
            let brake_heat = brake_pw[i] * abs(new_omega) * 0.000001 * dt_safe;
            let cooling    = 0.02 * (1.0 + speed * 0.01) * dt_safe;
            tire_temps[i]  = clamp(tire_temps[i] + slip_heat + brake_heat - cooling, 0.0, 1.0);
        }

        slip_ratios[i] = sr;
        slip_angles[i] = sa;
    }

    // Aplicar forças de pneu ao chassi
    forces  += forward_dir * total_long_force;
    forces  += -lateral_dir * total_lat_force;
    torques_y += total_yaw_torque;

    // yaw_inertia: fórmula da CPU (suposição morta documentada no cabeçalho)
    let yaw_inertia = c.mass * (half_wb * half_wb);

    // ============================================================
    // === FASE 4: COLISÃO COM PAREDES                          ===
    // ============================================================
    // A CPU (collision_system) modifica body.velocity e body.angular_velocity.y ANTES
    // da integration_system. Para replicar: usamos vel_col (inicia em c.vel, pre-integração)
    // e av_y_col (inicia em c.av_y). A integração na fase 5 parte dessas variáveis.
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
                // Impulso com restituição 0.05 (igual CPU linha 677)
                let impulse = -(1.0 + 0.05) * vel_into * c.mass;
                vel_col += pen_dir * (impulse / c.mass);
                // Atrito tangencial (CPU linha 682-683)
                let tang = vel_col - pen_dir * dot(vel_col, pen_dir);
                vel_col -= tang * 0.5;
                // Yaw kick: ADICIONADO DIRETAMENTE a av_y — igual à CPU (linha 687)
                // NÃO é torque; ignora yaw_inertia aqui (a CPU também não divide).
                let yaw_kick = cross(fwd, pen_dir).y;
                av_y_col += yaw_kick * vel_into * 0.05;
                // Empurrar para fora da parede
                pos_col -= pen_dir * 0.5;
            }
        }
    }

    // Ground clamp com heightmap (CPU linhas 696-700)
    let ground_y_c = height_at(c.pos.x, c.pos.z) + 0.5;
    if (pos_col.y < ground_y_c) {
        pos_col.y = ground_y_c;
        vel_col.y = max(vel_col.y, 0.0);
    }

    // ============================================================
    // === FASE 5: INTEGRAÇÃO (Euler Semi-Implícito)            ===
    // ============================================================
    // A CPU começa do vel_col (pós-colisão). Reaplica aero drag (DUPLICAÇÃO — ver cabeçalho).
    let speed_int = length(vel_col);
    var drag_force_int = vec3<f32>(0.0, 0.0, 0.0);
    if (flag(c.config_flags,2u)) {
        let drag_mag = 0.5 * RHO_AR * c.drag_coeff * FRONTAL_AREA * speed_int * speed_int;
        drag_force_int = -nor0(vel_col) * drag_mag;
    }

    let accel    = (forces + drag_force_int) / c.mass;
    var final_vel = vel_col + accel * dt_safe;

    // Lateral scrub friction (CPU linhas 727-735)
    let fwd_spd    = dot(final_vel, forward_dir);
    let lat_v      = final_vel - forward_dir * fwd_spd;
    if (length(lat_v) > 0.5) {
        let scrub = max(1.0 - 3.0 * dt_safe, 0.85);
        final_vel = forward_dir * fwd_spd + lat_v * scrub;
    }

    var final_pos = pos_col + final_vel * dt_safe;

    // Ground constraint mínimo (CPU linha 754-756)
    if (final_pos.y < 0.5) {
        final_pos.y = 0.5;
        final_vel.y = max(final_vel.y, 0.0);
    }

    // Yaw: damping escala com velocidade (CPU v10 — 400 + speed * 60)
    let base_damp  = 400.0;
    let speed_damp = speed_int * 60.0;
    // av_y_col inclui kicks diretos de colisão (já aplicados antes da integração, como na CPU)
    let drag_y  = -av_y_col * (base_damp + speed_damp);
    let yi      = select(c.mass * 2.0, yaw_inertia, yaw_inertia > 0.0);
    let ang_a_y = (torques_y + drag_y) / yi;
    let final_av_y = av_y_col + ang_a_y * dt_safe;

    // Rotação yaw: o host aplica transform.rotate_y(final_av_y * dt) depois de ler saida.

    // ============================================================
    // === ESCRITA DA SAÍDA                                     ===
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
