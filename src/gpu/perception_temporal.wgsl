// PERCEPÇÃO TEMPORAL — canais inter-tick (64-75, 81, 85-89, 99-113, 118-127, 192-207).
//
// # Por que este kernel existe separado de perception.wgsl
//
// O shader combinado (~1050 linhas) e a versão com loops de grade (287 linhas)
// ambos travavam o compilador ACO do RADV em `create_compute_pipeline` com SIGSEGV.
// O padrão específico que trava: loops `for` com `break` acessando `array<u32>` storage
// (a grade de voxels) dentro de um shader com workgroup_size=1.
//
// Solução: a CPU calcula as distâncias laterais e frontais usando os mesmos arrays
// de amostragem e injeta os 3 valores em `temporal[16-18]` antes do dispatch.
// O shader apenas lê esses campos e faz aritmética simples — sem nenhum loop de grade.
//
// # O que este kernel faz
//
// Lê o estado temporal por ser (binding 7: prev_dist, EMA, proximity_sustain, memory0,
// racing_line pré-computada, dist_l/r/f atuais pré-computados) e escreve nos canais
// temporais do buf_output (binding 6 — o MESMO buffer de saída do perception.wgsl).
//
// Os dois kernels escrevem em canais DISTINTOS:
//   perception.wgsl          → ch 0-97, 128-191
//   perception_temporal.wgsl → ch 64-75, 81, 85-89, 99-113, 118-127, 192-207
//     (99, 100, 121, 122, 124-127 vêm de update_extra2_temporal — CPU injeta,
//     shader só converte pra i8; 101 e 109 alimentados aqui — FASE CANAIS 2026-08-14,
//     ver comentários nos respectivos write_channel abaixo)
//
// # Fluxo no encoder
//
//   CPU: update_dist_temporal([dist_l, dist_r, dist_f] por ser → temporal[16-18])
//   CPU: update_racing_line_temporal([...] → temporal[10-15])
//   enc.clear_buffer(buf_output)
//   dispatch perception.wgsl             ← canais espaciais
//   dispatch perception_temporal.wgsl    ← canais temporais (sem loops de grade)
//   enc.copy_buffer_to_buffer(...)
//   queue.submit(enc)

// ─── Structs ───────────────────────────────────────────────────────────────────

struct Params {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    _pad:      f32,
}

struct Pose {
    pos_x: f32, pos_y: f32, pos_z: f32, phase: f32,
    fwd_x: f32, fwd_z: f32, rgt_x: f32, rgt_z: f32,
}

// ─── Bindings ──────────────────────────────────────────────────────────────────
// Idênticos ao layout temporal do perception.rs (8 bindings).
// Bindings 1-5 presentes mas NÃO acessados — necessários para o layout ser compatível
// com o bind group do perception.rs que reutiliza o mesmo layout_temporal.

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       grid:     array<u32>;      // não usado (loops removidos)
@group(0) @binding(2) var<storage, read>       flow:     array<vec2<f32>>; // não usado
@group(0) @binding(3) var<storage, read>       heights:  array<f32>;      // não usado
@group(0) @binding(4) var<storage, read>       poses:    array<Pose>;
@group(0) @binding(5) var<storage, read>       retina:   array<u32>;      // não usado
@group(0) @binding(6) var<storage, read_write> output:   array<atomic<u32>>;
// Estado temporal por ser — 58 f32 por ser (TEMPORAL_STRIDE=58).
// Layout:
//   [0-4]   prev_dist_l/r/f, wall_ema_l/r
//   [5]     proximity_sustain
//   [6]     joy_streak (reservado — slot histórico)
//   [7]     memory0
//   [8]     prev_line_offset_norm
//   [9]     _pad
//   [10-15] racing_line pré-computados pela CPU (ch108-110, ch118, ch119, ch123)
//   [16]    cpu_dist_l — dist lateral esq ATUAL (m), injetada pela CPU
//   [17]    cpu_dist_r — dist lateral dir ATUAL (m)
//   [18]    cpu_dist_f — dist frontal ATUAL (m)
//   [19]    _pad2
//   [20]    prop_alignment   — cos(heading vs pista) ∈ [-1,1] → ch64
//   [21]    prop_speed_m_s   — |vel| m/s ∈ [0,80]       → ch65 CRÍTICO (sabe se anda)
//   [22]    prop_lat_vel     — dot(vel,lat) m/s ∈ [-20,20] → ch66 (drift)
//   [23]    prop_hp_frac     — HP/HP_max ∈ [0,1]        → ch67
//   [24]    prop_rpm_ratio   — rpm/rpm_max ∈ [0,1]      → ch68
//   [25]    prop_gear        — marcha atual ∈ [0,7]      → ch69
//   [26]    prop_tire_temp_n — media_temp/120°C ∈ [0,1]  → ch70
//   [27]    prop_downforce_n — downforce/weight ∈ [0,1]  → ch71 (reservado=0)
//   [28]    prop_in_draft    — 1.0=em draft, 0.0=não      → ch72 (reservado=0)
//   [29]    prop_on_grass    — 1.0=grama, 0.0=pista       → ch73
//   [30]    prop_steer_abs   — |steer| ∈ [0,1]           → ch74
//   [31]    prop_wetness     — wetness ∈ [0,1]           → ch75 (reservado=0)
// ── NOVOS slots (v2) ─────────────────────────────────────────────────────────
//   [32]    reverse_timer    — tempo de ré > 0 enquanto manobra; decrementa → ch89
//   [33-48] cfc_hidden[0..15] — estado recorrente do CfC de percepção → ch192-207
//           Injetado pela CPU via update_cfc_percept_temporal() antes do dispatch.
//           Formula: wave[192+i] = existing + tanh(hidden[i]) * 127 * 0.3,
//           mas como ch192-207 começam zerados (clear_buffer), a fórmula simplifica
//           para: val = clamp(tanh(h) * 127.0 * 0.3, -127, 127).
// ── NOVOS slots (v3) — entregues via update_extra2_temporal() ────────────────
//   [49]    slip_signal   — cross2D(fwd_norm, vel_norm) [-1,1] → ch87
//   [50]    joy_tick      — progresso*superfície [-1,1] → ch126
//   [51]    joy_streak    — joy_streak_s/5.0 [0,1] → ch127
//   [52]    is_spinning   — 1.0 se |av_y|>1.5 && speed<2.0 → ch124 (-128)
//   [53]    comp_stress   — dist normalizada [0,1] → ch125
//   [54]    edge_row6     — edge density linha 6 retina [0,1] → ch99
//   [55]    edge_row7     — edge density linha 7 retina [0,1] → ch100
//   [56]    steer_mem     — phys.steer [-1,1] → ch121
//   [57]    brake_mem     — phys.brake_cmd [0,1] → ch122
@group(0) @binding(7) var<storage, read_write> temporal: array<f32>;

// ─── Constantes ────────────────────────────────────────────────────────────────

// TEMPORAL_STRIDE expandido de 50 → 58:
//   slots 0-48 existentes + 1 (slip_signal) + 4 (joy/spin/stress) + 4 (edge/steer/brake) = 58.
// CRÍTICO: deve bater exatamente com TEMPORAL_STRIDE em perception.rs.
// Slots 32-57 são injetados pela CPU via write_buffer antes do dispatch.
const TEMPORAL_STRIDE: u32 = 58u;
const WAVE_U32S:       u32 = 48u;  // ceil(192/4)

// ─── Utilitário: escreve canal i8 no buf_output via atomicOr ──────────────────
// Idêntico ao perception.wgsl.
fn write_channel(being: u32, ch: u32, val: i32) {
    let clamped  = clamp(val, -128, 127);
    let byte_val = u32(clamped & 0xFF);
    let base     = being * WAVE_U32S + ch / 4u;
    atomicOr(&output[base], byte_val << ((ch % 4u) * 8u));
}

// ─── Auxiliar: curvatura via campo de fluxo ───────────────────────────────────
//
// Usado para ch111 (350m), ch112 (200m) e ch113 (275m) — lookaheads de curvatura
// que estavam a cargo do motor.rs mas devem vir da GPU para evitar dependência
// de um código path que só roda no modo headless CPU.
//
// flow[] não é percorrido em loop — é apenas uma leitura direta por coordenada
// (nearest-neighbor), sem branch em cima de grid[], portanto não aciona o crash
// do compilador ACO que motivou a separação deste shader.
fn flow_at_temporal(wx: f32, wz: f32) -> vec2<f32> {
    let x = i32(round(wx / params.cell_size));
    let y = i32(round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if x >= 0 && x < n && y >= 0 && y < n {
        return flow[u32(y * n + x)];
    }
    return vec2<f32>(0.0, 0.0);
}

// Calcula curvatura (ângulo assinado entre forward e fluxo) em distância d.
// Resultado: [-π/2, π/2] → [-127, 127].
fn curv_at_d(pos_x: f32, pos_z: f32, fwd_x: f32, fwd_z: f32, d: f32) -> i32 {
    let probe_x = pos_x + fwd_x * d;
    let probe_z = pos_z + fwd_z * d;
    let fl = flow_at_temporal(probe_x, probe_z);
    let fmag2 = fl.x * fl.x + fl.y * fl.y;
    if fmag2 < 1e-4 {
        return 0;
    }
    let cross_y = fwd_x * fl.y - fwd_z * fl.x;
    let dot = clamp(fwd_x * fl.x + fwd_z * fl.y, -1.0, 1.0);
    let angle = atan2(cross_y, dot);
    let HALF_PI: f32 = 1.5707963267948966;
    return i32(clamp((angle / HALF_PI) * 127.0, -127.0, 127.0));
}

// ─── Kernel principal ──────────────────────────────────────────────────────────
//
// 1 workgroup por ser, 1 thread por workgroup.
// Não há loops de grade — apenas leituras de temporal[] e aritmética.
// O compilador ACO do RADV não trava com este padrão (travava com loops sobre storage).

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wgid: vec3<u32>) {
    let being = wgid.x;
    if being >= params.n_beings { return; }

    let tb = being * TEMPORAL_STRIDE;

    // ── Posição e direção deste ser (para lookaheads de curvatura) ────────────
    let pose  = poses[being];
    let pos_x = pose.pos_x;
    let pos_z = pose.pos_z;
    let fwd_x = pose.fwd_x;
    let fwd_z = pose.fwd_z;

    // ── Lê estado anterior ────────────────────────────────────────────────────
    let prev_dist_l  = temporal[tb + 0u];
    let prev_dist_r  = temporal[tb + 1u];
    let prev_dist_f  = temporal[tb + 2u];
    var ema_l        = temporal[tb + 3u];
    var ema_r        = temporal[tb + 4u];
    var prox_sustain = temporal[tb + 5u];
    let memory0      = temporal[tb + 7u];

    // ── Distâncias atuais pré-computadas pela CPU (campos [16-18]) ────────────
    // A CPU usa LAT_SAMPLES=[1,3,6,9,12,16,20,26,33,40]m e FWD_SAMPLES=[3..30]m para calcular
    // a menor distância a Wall/Hole em cada direção — o mesmo algoritmo que estava
    // nos loops que travavam o compilador ACO.
    let dist_l = temporal[tb + 16u];  // menor dist lateral esq (m), max=40
    let dist_r = temporal[tb + 17u];  // menor dist lateral dir (m), max=40
    let dist_f = temporal[tb + 18u];  // menor dist frontal (m), max=50

    // ── Approach rate (ch 102-107) ────────────────────────────────────────────
    // ar > 0 → parede se aproximando; ar < 0 → afastando.
    // DT = 1/64 s = 0.015625 s.
    let DT: f32 = 0.015625;
    let ar_l = (prev_dist_l - dist_l) / DT;
    let ar_r = (prev_dist_r - dist_r) / DT;
    let ar_f = (prev_dist_f - dist_f) / DT;

    // ch 102-104: approach rate lateral esq/dir e frontal (±30m/s → ±127)
    write_channel(being, 102u, i32(clamp(ar_l / 30.0 * 127.0, -127.0, 127.0)));
    write_channel(being, 103u, i32(clamp(ar_r / 30.0 * 127.0, -127.0, 127.0)));
    write_channel(being, 104u, i32(clamp(ar_f / 30.0 * 127.0, -127.0, 127.0)));

    // ch 105: TTC mínimo (tempo até colisão, 127=iminente, 0=seguro)
    let ttc_l   = select(2.0, clamp(dist_l / max(ar_l, 0.001), 0.0, 2.0), ar_l > 1.0);
    let ttc_r   = select(2.0, clamp(dist_r / max(ar_r, 0.001), 0.0, 2.0), ar_r > 1.0);
    let ttc_f   = select(2.0, clamp(dist_f / max(ar_f, 0.001), 0.0, 2.0), ar_f > 1.0);
    let ttc_min = clamp(min(min(ttc_l, ttc_r), ttc_f), 0.0, 2.0);
    write_channel(being, 105u, i32(127.0 * (1.0 - ttc_min / 2.0)));

    // ch 106-107: delta-dist lateral ±15m/s (ganho 64/15 ≈ 4.3×)
    write_channel(being, 106u, i32(clamp(ar_l / 15.0 * 64.0, -127.0, 127.0)));
    write_channel(being, 107u, i32(clamp(ar_r / 15.0 * 64.0, -127.0, 127.0)));

    // ── Wall EMA (ch 85, 86) ──────────────────────────────────────────────────
    // alpha = dt*3 ≈ 0.047 → meia-vida ~21 ticks (~0.33s).
    let ema_alpha = clamp(DT * 3.0, 0.0, 1.0);
    ema_l = ema_l * (1.0 - ema_alpha) + dist_l * ema_alpha;
    ema_r = ema_r * (1.0 - ema_alpha) + dist_r * ema_alpha;
    // Invertido: 127 = parede encostando (≤0m), 0 = livre (≥40m).
    // Normaliza por 40.0 (alcance máximo das amostras laterais expandido).
    write_channel(being, 85u, i32(clamp((1.0 - clamp(ema_l / 40.0, 0.0, 1.0)) * 127.0, 0.0, 127.0)));
    write_channel(being, 86u, i32(clamp((1.0 - clamp(ema_r / 40.0, 0.0, 1.0)) * 127.0, 0.0, 127.0)));

    // ── Proximity sustain (ch 88) ─────────────────────────────────────────────
    // Acumula enquanto ema_l ou ema_r < 3m (raspando lateral).
    let raspando = ema_l < 3.0 || ema_r < 3.0;
    if raspando {
        prox_sustain = clamp(prox_sustain + DT, 0.0, 3.0);
    } else {
        prox_sustain = clamp(prox_sustain - DT * 0.5, 0.0, 3.0);
    }
    write_channel(being, 88u, i32(clamp(prox_sustain / 3.0 * 127.0, 0.0, 127.0)));

    // ── Lane offset (ch 81) — assimetria lateral na pista ────────────────────
    // lane_offset = (dist_r - dist_l) / (dist_l + dist_r), [-1..+1].
    // Positivo = mais espaço à direita (ser está à esquerda da pista).
    // Estava definido apenas em motor.rs, nunca alimentado pela GPU — agora
    // calculado aqui usando dist_l/r já disponíveis nos campos temporais [16-17].
    let lane_denom = (dist_l + dist_r);
    if lane_denom > 0.1 {
        let lane_off = (dist_r - dist_l) / lane_denom;
        write_channel(being, 81u, i32(clamp(lane_off * 127.0, -127.0, 127.0)));
    } else {
        write_channel(being, 81u, 0);
    }

    // ── Slip angle (ch 87) — mantido zero aqui ────────────────────────────────
    // ch87 requer angular_velocity do physics body, disponível apenas na CPU
    // (motor.rs:1614). O motor.rs pode sobrescrever ch87 após GPU perception.
    // Aqui garante-se que não fica lixo de tick anterior (buf_output é zerado).
    // Não é necessário write_channel(87, 0) porque clear_buffer garante zero.

    // ── Memória (ch 120) ──────────────────────────────────────────────────────
    // throttle intencional do tick anterior [-1,1] → [-127,127]
    write_channel(being, 120u, i32(clamp(memory0 * 127.0, -127.0, 127.0)));

    // ── Racing line (ch 108-110, 118, 119, 123) pré-computados pela CPU ───────
    // CPU injeta em temporal[10-15] antes do dispatch via update_racing_line_temporal().
    // temporal[11] (antigo "cpu_d_off") NÃO é mais lido aqui — a CPU sempre enviou
    // 0.0 nesse slot (derivada nunca implementada no host, ver histórico de
    // Channel::LineOffsetDeriv em channels.rs). O slot continua sendo escrito pela
    // CPU só para não mudar o layout de TEMPORAL_STRIDE; vestigial, sem leitor.
    let cpu_line_off_norm = temporal[tb + 10u];
    let cpu_la_off        = temporal[tb + 12u];
    let cpu_fwd_norm      = temporal[tb + 13u];
    let cpu_corridor      = temporal[tb + 14u];
    let cpu_whisker_norm  = temporal[tb + 15u];

    write_channel(being, 108u, i32(clamp(cpu_line_off_norm *  127.0, -127.0, 127.0)));

    // ch109: derivada REAL do desvio da racing line — FIX 2026-08-14 (FASE CANAIS).
    // ANTES: hardcoded 0.0 (a CPU nunca implementou o cálculo — ver comentário acima).
    // O slot [8] deste MESMO buffer já guarda `cpu_line_off_norm` do tick ANTERIOR
    // (salvo por este shader na seção "Atualiza estado temporal" no fim do arquivo) —
    // a derivada é computável aqui, sem precisar de mais um slot injetado pela CPU.
    // Vira TAXA (offset normalizado / segundo, dividido por DT) para não depender da
    // frequência do tick — mesmo padrão de ApproachRate (ch102-104) acima.
    // MAX_OFFSET_RATE=8,0/s é ESTIMATIVA (corredor típico atravessado em <1s na
    // velocidade máxima observada, ~28 m/s) — NÃO validada por medição (GPU ocupada
    // com baterias em 2026-08-14); calibrar com --auditar-visao antes de fechar a fase.
    let prev_line_off_norm = temporal[tb + 8u];
    let d_off_rate = (cpu_line_off_norm - prev_line_off_norm) / DT;
    let MAX_OFFSET_RATE: f32 = 8.0;
    write_channel(being, 109u, i32(clamp((d_off_rate / MAX_OFFSET_RATE) * 127.0, -127.0, 127.0)));

    // ch110: lookahead offset, sinal invertido (afastando da linha ideal → virar de volta)
    write_channel(being, 110u, i32(clamp(cpu_la_off        * -127.0, -127.0, 127.0)));
    write_channel(being, 118u, i32(clamp(cpu_fwd_norm      *  127.0,    0.0, 127.0)));
    write_channel(being, 119u, i32(clamp(cpu_corridor      *  127.0,    0.0, 127.0)));
    write_channel(being, 123u, i32(clamp(cpu_whisker_norm  *  127.0,    0.0, 127.0)));

    // ── Lookaheads de curvatura horizonte médio-longo (ch 111-113) ──────────────
    // ch111 estava a cargo do motor.rs (linha 1782: sensory_wave[111] = c350) mas só
    // era escrito no path CPU/headless — a GPU nunca o alimentava. Movido para cá
    // para garantir alimentação consistente a qualquer tick.
    // ch112 (200m) e ch113 (275m) eram slots zero sem fonte; preenchidos agora com
    // curvatura em horizontes intermediários que complementam ch79 (70m) e ch80 (120m).
    write_channel(being, 111u, curv_at_d(pos_x, pos_z, fwd_x, fwd_z, 350.0));
    write_channel(being, 112u, curv_at_d(pos_x, pos_z, fwd_x, fwd_z, 200.0));
    write_channel(being, 113u, curv_at_d(pos_x, pos_z, fwd_x, fwd_z, 275.0));

    // ── ch101: curvatura a 260m — FIX 2026-08-14 (FASE CANAIS) ────────────────
    // ANTES: só motor.rs escrevia (`sensory_wave[101] = c260`, caminho Bevy CPU) —
    // `gpu_evolve` (o binário de treino) não importa motor.rs, então este canal
    // ficava sem fonte nenhuma nesse caminho. Reusa `curv_at_d`, a MESMA função já
    // validada para ch111-113 (só muda o argumento de distância) — sem lógica nova.
    write_channel(being, 101u, curv_at_d(pos_x, pos_z, fwd_x, fwd_z, 260.0));

    // ── Telemetria proprioceptiva (ch 64-75) — injetada via temporal[20-31] ──────
    //
    // Sem esses canais o modelo não sabe se está em movimento (ch65=speed) ou
    // se o motor está acelerado (ch68=RPM), o que resulta em throttle≈0 perpétuo.
    // O CHAMPION.dna tem pesos L1 significativos em ch65 (mag≈23), ch64 (alignment),
    // ch69 (gear) e ch70 (tire_temp) — confirmar inspecionando SRLE tag do DNA.
    //
    // Codificação: idêntica ao motor.rs (simulation/motor.rs linhas 1189-1300).
    //   ch64: alignment [-1,1] → *127 → i8
    //   ch65: speed_m_s [0,80] → *2.0 → i8 (127 = 63.5 m/s ≈ 229 km/h)
    //   ch66: lat_vel [-20,20] → *10.0 → i8
    //   ch67: hp_frac [0,1]   → *127 → i8
    //   ch68: rpm_ratio [0,1] → *127 → i8
    //   ch69: gear [0,7]      → gear*20 → i8 (127 = marcha 6+)
    //   ch70: tire_temp [0,1] → *127 → i8
    //   ch71: temperatura do motor [20-150°C → 0-1 → *127] (slot era downforce, sobrescrito)
    //   ch72: temperatura do óleo [idem] (slot era draft, sobrescrito no mesmo commit)
    //   ch73: on_grass 0/1    → -128 se grama (dor), 0 se pista
    //   ch74: steer_limit [0,1] → *127 → i8
    //   ch75: estagnação [0-64 ticks → 0-1 → *127] (slot era wetness, sobrescrito)

    let prop_alignment   = temporal[tb + 20u];  // cos(heading vs pista) [-1,1]
    let prop_speed_m_s   = temporal[tb + 21u];  // |vel| m/s [0,80]
    let prop_lat_vel     = temporal[tb + 22u];  // dot(vel,lat) m/s [-20,20]
    let prop_hp_frac     = temporal[tb + 23u];  // HP/HP_max [0,1]
    let prop_rpm_ratio   = temporal[tb + 24u];  // rpm/rpm_max [0,1]
    let prop_gear        = temporal[tb + 25u];  // marcha atual [0,7]
    let prop_tire_temp   = temporal[tb + 26u];  // media_temp/120°C [0,1]
    let prop_downforce   = temporal[tb + 27u];  // downforce/weight [0,1]
    let prop_in_draft    = temporal[tb + 28u];  // 0.0 ou 1.0
    let prop_on_grass    = temporal[tb + 29u];  // 0.0 ou 1.0
    let prop_steer_abs   = temporal[tb + 30u];  // |steer| [0,1]
    let prop_wetness     = temporal[tb + 31u];  // wetness [0,1]

    // ch64: alignment — 127 = perfeitamente alinhado, 0 = perpendicular, -127 = contra-mão
    write_channel(being, 64u, i32(clamp(prop_alignment * 127.0, -127.0, 127.0)));
    // ch65: speed — 127 = 63.5 m/s (228 km/h). CRÍTICO: modelo sabe se está andando.
    write_channel(being, 65u, i32(clamp(prop_speed_m_s * 2.0, 0.0, 127.0)));
    // ch66: lateral velocity (drift) — positivo = deslizando para direita
    write_channel(being, 66u, i32(clamp(prop_lat_vel * 10.0, -127.0, 127.0)));
    // ch67: HP restante — 127 = cheio, 0 = morto
    write_channel(being, 67u, i32(clamp(prop_hp_frac * 127.0, 0.0, 127.0)));
    // ch68: RPM ratio — 127 = redline
    write_channel(being, 68u, i32(clamp(prop_rpm_ratio * 127.0, 0.0, 127.0)));
    // ch69: marcha — 20 por marcha (marcha 6 → 120, max ~127)
    write_channel(being, 69u, i32(clamp(prop_gear * 20.0, 0.0, 127.0)));
    // ch70: temperatura dos pneus — 127 = 120°C (ótima)
    write_channel(being, 70u, i32(clamp(prop_tire_temp * 127.0, 0.0, 127.0)));
    // ch71: temperatura do motor — o slot era downforce mas foi sobrescrito; a CPU injeta
    //       temp_motor via temporal[27]. Acima de 105°C a potência cai (physics.wgsl:402).
    write_channel(being, 71u, i32(clamp(prop_downforce * 127.0, 0.0, 127.0)));
    // ch72: temperatura do óleo — slot era in_draft, sobrescrito pela CPU com temp_oleo
    //       via temporal[28]. Mais lento que o motor (τ ≈ 500-1000 ticks).
    write_channel(being, 72u, i32(clamp(prop_in_draft * 127.0, 0.0, 127.0)));
    // ch73: dor de grama — -128 = grama (penalidade severa), 0 = pista normal
    write_channel(being, 73u, select(0, -128, prop_on_grass > 0.5));
    // ch74: limite de esterçamento — 127 = tudo steerable, 0 = travado
    write_channel(being, 74u, i32(clamp(prop_steer_abs * 127.0, 0.0, 127.0)));
    // ch75: estagnação — slot era wetness, sobrescrito pela CPU com stagnation_ticks/64
    //       via temporal[31]. 127 = 64 ticks sem avançar (candidato a morte por parede).
    write_channel(being, 75u, i32(clamp(prop_wetness * 127.0, 0.0, 127.0)));

    // ── reverse_timer (ch 89) — injetado via temporal[32] ────────────────────
    // reverse_timer > 0 enquanto em manobra de ré de emergência.
    // A CPU decrementa e grava em temporal[32] antes do dispatch via update_extra_temporal().
    // ch89: 127 = em ré, 0 = marcha à frente. Binary: qualquer valor > 0 → 127.
    // Weight L1 no CHAMPION.dna: parte do sinal de contexto de marcha.
    let rev_timer = temporal[tb + 32u];
    let is_reversing = rev_timer > 0.0;
    write_channel(being, 89u, select(0, 127, is_reversing));

    // ── CfC de percepção (ch 192-207) — estado recorrente injetado pela CPU ──
    // O CfC de percepção (RetinaPerception em perception.rs) processa a retina
    // com um pipeline CfC(16) e produz hidden[0..15].
    // Esses valores são injetados em temporal[33-48] pela CPU antes do dispatch
    // via update_cfc_percept_temporal().
    //
    // Formula: wave[192+i] = clamp(tanh(hidden[i]) * 127.0 * 0.3, -127, 127)
    // Strength 0.3 = mesmo valor do motor.rs L1626-1628.
    // Como buf_output foi zerado (clear_buffer), existing = 0, então simplifica:
    //   val = tanh(h) * 38.1   (127 × 0.3 ≈ 38.1)
    //
    // Peso L1 total dos canais 192-207 no CHAMPION.dna = 7.1% — o maior item faltante.
    // Sem esses canais o modelo não tem contexto temporal visual de longo prazo.
    for (var ci = 0u; ci < 16u; ci++) {
        let h = temporal[tb + 33u + ci];
        // tanh aproximado: x/(1+|x|) tem erro < 2% para |x|<3, suficiente para i8.
        let abs_h = abs(h);
        let tanh_h = select(h / (1.0 + abs_h), sign(h) * 0.999, abs_h > 10.0);
        let val = i32(clamp(tanh_h * 127.0 * 0.3, -127.0, 127.0));
        write_channel(being, 192u + ci, val);
    }

    // ── Canais dopaminérgicos e de stress (slots v3, injetados pela CPU) ─────
    // Os valores chegam via update_extra2_temporal(), calculados na CPU com dados
    // do loop de odômetro e da física. O shader só os lê e converte para i8.

    // ch87: slip angle — cross2D(fwd, vel_norm) [-1,1]. sem fonte no GPU = injetado via CPU.
    // Motor.rs:1614 usa a mesma fórmula (cross product 2D entre fwd e vel normalize).
    let slip_signal = temporal[tb + 49u];
    write_channel(being, 87u, i32(clamp(slip_signal * 127.0, -127.0, 127.0)));

    // ch126: joy_tick — progresso*superfície [-1,1]. Mesmo cálculo de motor.rs:1673.
    // Positivo = avançando na pista, negativo = colidindo/capim.
    let joy_tick = temporal[tb + 50u];
    write_channel(being, 126u, i32(clamp(joy_tick * 127.0, -127.0, 127.0)));

    // ch127: joy_streak — segundos de flow / 5.0 [0,1]. Motor.rs:1684.
    let joy_streak_n = temporal[tb + 51u];
    write_channel(being, 127u, i32(clamp(joy_streak_n * 127.0, 0.0, 127.0)));

    // ch124: dor de rodopio — -128 se girando (|av_y|>1.5 && speed<2.0). Motor.rs:1384.
    let is_spin = temporal[tb + 52u];
    write_channel(being, 124u, select(0, -128, is_spin > 0.5));

    // ch125: estresse de proximidade — 64 se atrás, 127 se alguém a <20m à frente. Motor.rs:1379.
    // Escala: 0→líder, 0.5→meio pelotão, 1.0→alguém próximo.
    let comp_stress_n = temporal[tb + 53u];
    write_channel(being, 125u, i32(clamp(comp_stress_n * 127.0, 0.0, 127.0)));

    // ch99: edge density linha 6 da retina [0,1]. Motor.rs:923 ("edge_norm").
    // Disponível no host via readback; calculado com a mesma fórmula.
    let edge6 = temporal[tb + 54u];
    write_channel(being, 99u, i32(clamp(edge6 * 127.0, 0.0, 127.0)));

    // ch100: edge density linha 7 da retina [0,1]. Motor.rs:923.
    let edge7 = temporal[tb + 55u];
    write_channel(being, 100u, i32(clamp(edge7 * 127.0, 0.0, 127.0)));

    // ch121: steer_mem = phys.steer suavizado [-1,1]. Motor.rs:1660 (racer.memory[1]).
    // No caminho GPU o análogo é phys.steer — o volante que a física recebeu.
    let steer_mem = temporal[tb + 56u];
    write_channel(being, 121u, i32(clamp(steer_mem * 127.0, -127.0, 127.0)));

    // ch122: brake_mem = phys.brake_cmd [0,1]. Motor.rs:1661 (racer.memory[2]).
    let brake_mem = temporal[tb + 57u];
    write_channel(being, 122u, i32(clamp(brake_mem * 127.0, 0.0, 127.0)));

    // ── Atualiza estado temporal para o próximo tick ──────────────────────────
    // Salva as distâncias ATUAIS como "anteriores" para o próximo approach_rate.
    temporal[tb + 0u] = dist_l;
    temporal[tb + 1u] = dist_r;
    temporal[tb + 2u] = dist_f;
    temporal[tb + 3u] = ema_l;
    temporal[tb + 4u] = ema_r;
    temporal[tb + 5u] = prox_sustain;
    // [6] joy_streak: mantém
    // [7] memory0: atualizado pela CPU via update_memory0() após física
    // [8] prev_line_offset_norm: atualiza com valor atual (para d_off do próximo tick)
    temporal[tb + 8u] = cpu_line_off_norm;
    // [9-19]: campos CPU — sobrescritos pela CPU antes do próximo dispatch
    // [32]: reverse_timer — decrementado e reinjetado pela CPU
    // [33-48]: cfc_hidden — reinjetado pela CPU todo tick (após rodar CfC com retina)
}
