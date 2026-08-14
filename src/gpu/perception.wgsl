// PERCEPÇÃO NA GPU — redução de 1764 pixels retina → 223 canais sensory_wave.
//
// # Por que este kernel existe
//
// A retina já roda na GPU (retina.wgsl) e entrega distância + material por pixel (FÍSICA),
// mas os 1764 pixels voltavam à CPU para serem reduzidos canal a canal no motor.rs.
// Essa roundtrip custava a maior transferência do laço: 1764 B × N seres por tick.
// Com 60 seres: 1764 × 60 × 2 (ida e volta) = ~211 KB/tick × 64 Hz = ~13,5 MB/s de PCIe
// que deixava de servir o trabalho real da GPU.
//
// Este kernel consome a saída da retina DIRETAMENTE em VRAM (buf_retina_fisica) e escreve
// o sensory_wave (223 i8 por ser) num buffer separado — sem tocar a CPU no caminho quente.
//
// # O que é portável e o que fica na CPU
//
// Portável (local ao ser, sem dependência inter-seres):
//   ch  0- 7  FAR pool (rows 0-4, 8 bins, max-pool) — sobre codificação legada
//   ch  8-15  NEAR pool (rows 10-13, 8 bins, max-pool) — idem
//   ch 16-23  wall_distance_fov (8 ângulos YAW, consulta grade, range 60m)
//   ch 24-31  wall_width_fov (largura por ângulo, 15m à frente)
//   ch 32-39  wall_curvature_fov (cross product flow vs forward)
//   ch 40-47  road_ahead NEAR (5..40m)
//   ch 48-55  road_ahead FAR (45..150m)
//   ch 56-59  material atual (4×)
//   ch 60-63  escape direction (run de road mais longo no setor frontal)
//   ch 76-77  dist lateral esq/dir (amostras a 1-40m, BUG 1 fix 2026-08-04 + expansão 40m)
//   ch 78-82  curvatura lookahead + urgência frontal
//   ch 90-93  dark-center tracker (sliding window sobre row RETINA_ROWS/2−1)
//   ch 94-97  whiskers diagonais (±30°, ±60°)
//   ch 128-143 Fóvea HD (16 bins, max todas as rows)
//   ch 144-159 Periférica HD (16 bins, max todas as rows)
//   ch 160-191 Horizon scan raw (row RETINA_ROWS/2, 32 amostras)
//
// NÃO portado NESTE ARQUIVO (mantém fora de perception.wgsl — leia comentários em
// percepcao.rs). ATENÇÃO: esta lista é de quando só existia este kernel; a maioria
// destes canais JÁ tem fonte GPU hoje, só que em OUTRO arquivo — perception_temporal
// .wgsl e perception_motion.wgsl, ambos despachados no MESMO CommandEncoder (ver
// header desses dois arquivos para a lista atual, canal por canal):
//   ch 21-27, 98    motion per-row — GPU: perception_motion.wgsl (não CPU)
//   ch 29-31, 33-35 edge density por row (idem)
//   ch 44-51, 60-63 VÁRIOS: o motor.rs sobrescreve esses slots em ordens diferentes —
//                   veja nota "conflito de slots" abaixo
//   ch 64-75  telemetria proprioceptiva (RPM, velocidade, temperatura, etc. — estado do corpo)
//   ch 72     drafting (hash espacial entre seres — inter-ser)
//   ch 83-84  fluxo óptico hemisférico — GPU: perception_motion.wgsl (não CPU)
//   ch 114-117 approach hemisférico + fóvea — GPU: perception_motion.wgsl desde
//              FASE CANAIS (2026-08-14); antes só motor.rs (Bevy CPU) alimentava
//              no binário de treino gpu_evolve — ver Channel::ApproachHemiL em
//              perception/channels.rs para a justificativa completa.
//   ch 85-86  wall EMA (acumulado por ser entre ticks)
//   ch 87     slip angle (estado físico do corpo)
//   ch 88     proximity_sustain (acumulado por ser)
//   ch 89     estado de ré (acumulado)
//   ch 99-100 edge rows 7+
//   ch 102-107 approach rate (requer wall_dist_prev — estado temporal)
//   ch 108-110 racing line offset (estado temporal + racing_line.half_width)
//   ch 111    curvatura 350m (calculado com forward — requer estado do corpo)
//   ch 118-124 width-invariant, whisker_min (requerem racing_line.half_width)
//   ch 125    estresse competitivo (inter-ser, hash espacial)
//   ch 126-127 dopamina / joy streak (acumulados)
//   ch 192+   injeção CfC de percepção (recorrente, estado temporal por ser)
//
// # Nota: conflito de slots no motor.rs
//
// O motor.rs reescreve alguns slots várias vezes em ordem diferente da documentação.
// Por exemplo, ch 36-43 é preenchido por open_l (36-39) e open_r (40-43), DEPOIS
// ch 40-47 (road_ahead NEAR) é escrito por cima. E escape_i8 vai para 60-63, mas
// depois ch 60-63 do bloco "agent_state" coloca 0 — o escape sobrescreve porque vem
// DEPOIS. Esta GPU usa a semântica FINAL de cada slot (última escritura). Canais onde
// há conflito são marcados com ATENÇÃO abaixo.
//
// # Mudança de sinal: de legado para física
//
// A CPU usa retina_binocular que é a CODIFICAÇÃO LEGADA (i8, road=-128..-100, wall=53..127).
// Este kernel tem acesso à FÍSICA (distância m + material).
//
// Para os canais de pooling (ch 0-15, 128-191) que fazem max-pool sobre a codificação legada,
// RECONSTRUÍMOS a codificação legada pixel a pixel via `legacy_encoding()` e fazemos
// max-pool sobre ela — equivalente à CPU. Isto preserva a semântica do motor.rs (Wall domina,
// Road desaparece em bins com parede). A alternativa (física pura: distância mínima por bin)
// mudaria o sinal e exigiria revalidação da evolução — decisão adiada para integração D.
//
// Para os canais semânticos (ch 16-63, 76-82, 90-97) que consultam a GRADE diretamente,
// a GPU tem acesso à grade — esses são EQUIVALENTES.
//
// # Tolerância esperada
//
// Canais que consultam a grade: equivalência exata (mesma função `round(wx/cell_size)`).
// Canais de pooling (ch 0-15, 128-191): DIFEREM da CPU porque a CPU usa codificação legada
// e este kernel usa física. Isso deve ser REVALIDADO após integração (ver percepcao.rs).

// ─── Structs de entrada ────────────────────────────────────────────────────────

struct Params {
    grid_size:  u32,    // 320
    n_beings:   u32,    // 60
    cell_size:  f32,    // 3.0 m
    _pad:       f32,
}

// Pose de um ser no tick atual.
struct Pose {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    phase: f32,       // racer.phase (heading absoluto em radianos, usado nos whiskers)
    fwd_x: f32,
    fwd_z: f32,       // forward_2d (normalizado)
    rgt_x: f32,
    rgt_z: f32,       // right_2d (normalizado)
}

// ─── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>          params:      Params;
// grade de voxels — u32, mesmo layout de retina.wgsl
@group(0) @binding(1) var<storage, read>    grid:        array<u32>;
// campo de fluxo — vec2<f32> por célula, stride = grid_size
@group(0) @binding(2) var<storage, read>    flow:        array<vec2<f32>>;
// mapa de altura — f32 por célula, bilinear no host; aqui lemos nearest (piso flat)
// ATENÇÃO: get_height_at usa interpolação bilinear na CPU. Aqui usamos nearest-neighbor
// (mais simples, sem branch). O erro máximo é proporcional ao gradiente de altura —
// numa pista plana é zero; em rampa com slope 0.1 m/m e cell_size 3m o erro é ≤0.3m.
@group(0) @binding(3) var<storage, read>    heights:     array<f32>;
// poses dos seres — 1 por ser
@group(0) @binding(4) var<storage, read>    poses:       array<Pose>;
// saída da retina física: packed u32 = (mm << 8) | material, 1764 por ser
@group(0) @binding(5) var<storage, read>    retina:      array<u32>;
// saída: sensory_wave de 223 i8 por ser, empacotado em u32 atômico (56 u32, byte 0-222 usados)
// Layout: wave_u32[ser * 56 + word] >> (byte*8) & 0xFF, interpretado como i8.
// Atomic é NECESSÁRIO: threads distintos do mesmo workgroup podem escrever bytes em
// palavras compartilhadas (ex.: thread 58 escreve ch 76-77 = word 19, e thread 59 escreve
// ch 78-79 = word 19). Sem atomicOr, a leitura-modificação-escrita provoca race condition
// silenciosa e valores errados em qualquer canal onde ch/4 colidir entre threads.
@group(0) @binding(6) var<storage, read_write> output:  array<atomic<u32>>;

// Binding 7 (temporal state) MOVIDO para perception_temporal.wgsl.
// Este kernel (perception.wgsl) processa apenas os canais espaciais/retina (ch 0-97, 128-191).
// O kernel perception_temporal.wgsl roda em seguida no MESMO encoder e escreve os
// canais temporais (85, 86, 88, 102-110, 118-120, 123) no MESMO buf_output.
// Encadear os dois no mesmo CommandEncoder garante que perception_temporal vê o
// buf_output zerado e escreve nos slots livres sem race condition.

// ─── Constantes ────────────────────────────────────────────────────────────────

const RETINA_ROWS:    u32 = 14u;
const RETINA_COLS:    u32 = 126u;
const RETINA_TOTAL:   u32 = 1764u;   // 14 × 126
const WAVE_SIZE:      u32 = 192u;   // = SENSORY_WAVE_SIZE em config.rs
const WAVE_U32S:      u32 = 48u;     // ceil(192 / 4) = 48

// Amostras para sensores laterais (ch 76-77): expandidas de 27m → 40m.
// CONTA DO ALCANCE: velocidade máxima observada = 28 m/s.
// A 28 m/s o ser percorre 40 m em 40/28 = 1.43 s — antecipação confortável.
// Com 27m o prazo era 27/28 = 0.96 s (marginal). Com 40m = 1.43 s (≥1 s seguro).
// O frontal usa 30 m (1.07 s) — o lateral deve cobrir pelo menos o mesmo horizonte.
// BUG 1 (2026-08-04): 56% dos seres travavam por parede LATERAL sem ver a ameaça.
// Fix: alcance lateral elevado a 40m, 10 amostras, normaliza por 40m.
const LAT_SAMPLES: array<f32, 10> = array(1.0, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 33.0, 40.0);

// Amostras para whiskers (ch 94-97): mesmas do motor.rs
const W_SAMPLES: array<f32, 5> = array(1.0, 3.0, 5.0, 7.0, 10.0);

// ATENÇÃO: o motor.rs usa um angulo de 50° pra o yaw sweep dos sensores 16-39.
const FOV_HALF_RAD: f32 = 0.872664625997; // 50° em radianos = pi/3.6

// ─── Utilitários de grade ───────────────────────────────────────────────────────

// Retorna o tipo de voxel em coordenadas mundo. Idêntico a get_voxel_at() em world.rs:
//   round(wx / cell_size) = índice de célula.
// Fora da grade → Wall (u32 = 2), igual à CPU.
fn voxel_at(wx: f32, wz: f32) -> u32 {
    let x = i32(round(wx / params.cell_size));
    let y = i32(round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if x >= 0 && x < n && y >= 0 && y < n {
        return grid[u32(y * n + x)];
    }
    return 2u; // Wall fora da grade
}

// Retorna fluxo em coordenadas mundo. Idêntico a get_flow_at(): round nearest.
fn flow_at(wx: f32, wz: f32) -> vec2<f32> {
    let x = i32(round(wx / params.cell_size));
    let y = i32(round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if x >= 0 && x < n && y >= 0 && y < n {
        return flow[u32(y * n + x)];
    }
    return vec2<f32>(0.0, 0.0);
}

// Retorna altura em coordenadas mundo. Usa nearest-neighbor em vez de bilinear.
// Erro ≤ cell_size × slope_max. Numa pista plana: erro = 0. Aceitável para ch 48-55.
fn height_at(wx: f32, wz: f32) -> f32 {
    let x = i32(round(wx / params.cell_size));
    let y = i32(round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if x >= 0 && x < n && y >= 0 && y < n {
        return heights[u32(y * n + x)];
    }
    return 0.0;
}

// ─── Empacotamento da saída ─────────────────────────────────────────────────────

// A saída são 223 bytes (i8 interpretados como u8 sem sinal para empacotamento).
// Mapeamento: wave[ser][ch] → saida[(ser * WAVE_U32S) + (ch / 4)], byte (ch % 4).
// Cada u32 armazena 4 bytes em ordem little-endian (byte 0 = bits 7:0).
//
// ATENÇÃO sobre atomics: vários threads do mesmo workgroup podem escrever palavras
// diferentes dentro do mesmo bloco. Usamos `workgroup_size(64)` com 1 workgroup por ser,
// então cada thread escreve palavras distintas sem conflito — NÃO precisamos de atomics.
// Se mudar para N seres compartilhando um workgroup, rever.

fn write_channel(being: u32, ch: u32, val: i32) {
    // Clamp em [-128, 127] — igual ao cast `as i8` em Rust.
    let clamped = clamp(val, -128, 127);
    // Reinterpreta como u8 sem sinal (two's complement: -128 → 0x80, -1 → 0xFF).
    let byte_val = u32(clamped & 0xFF);
    let word_idx = ch / 4u;
    let byte_pos = ch % 4u;
    let shift = byte_pos * 8u;
    let base = being * WAVE_U32S + word_idx;
    // atomicOr: escreve apenas os 8 bits do byte alvo sem perturbar os outros bytes da palavra.
    // PRÉ-CONDIÇÃO: o host DEVE zerar buf_saida com enc.clear_buffer() antes de cada dispatch.
    // Zero inicial + OR por byte = valor final correto. Sem zerar: bits do tick anterior persistem.
    // O host (percepcao.rs::reduzir) garante enc.clear_buffer(&buf_saida, 0, None) antes do dispatch.
    atomicOr(&output[base], byte_val << shift);
}

fn read_retina_pixel(being: u32, row: u32, col: u32) -> u32 {
    // packed = (mm << 8) | material
    return retina[being * RETINA_TOTAL + row * RETINA_COLS + col];
}

// Distância em metros de um pixel da retina.
fn pixel_dist_m(packed: u32) -> f32 {
    return f32(packed >> 8u) / 1000.0;
}

// Material de um pixel (0=Grass, 1=Road, 2=Wall, 3=Hole).
fn pixel_mat(packed: u32) -> u32 {
    return packed & 0xFFu;
}

// Codificação LEGADA a partir de FÍSICA — replica legacy_encoding() em mod.rs.
//
// CONTEXTO: canais ch 0-15 e 128-191 na CPU fazem max-pool sobre a CODIFICAÇÃO LEGADA.
// A codificação legada mapeia: Wall perto → +127, Road → -128..−100, Grass → ~0..+40.
// Com max-pool, uma célula com Wall domina toda a bin, apagando Road.
//
// Na GPU temos FÍSICA (distância + material separados). Para ser EQUIVALENTE à CPU,
// reconstruímos a codificação legada aqui e fazemos max-pool sobre ela.
//
// ALTERNATIVA FÍSICA (documentada): usar distância mínima por bin + fração de material.
// Isso MUDA o sinal e requer revalidação da evolução. Por isso optamos por equivalência
// via legado — a decisão de migrar para física pura é da integração final (D).
fn legacy_encoding(d: f32, mat: u32) -> i32 {
    let depth_fade = clamp(d / 1200.0, 0.0, 1.0);
    var v: f32;
    if mat == 2u { // Wall
        if d < 30.0 {
            v = 127.0 - (sqrt(clamp(d / 30.0, 0.0, 1.0)) * 72.0);
        } else if d < 120.0 {
            v = 55.0 - (clamp((d - 30.0) / 90.0, 0.0, 1.0) * 55.0);
        } else if d < 1200.0 {
            v = 0.0 - (clamp((d - 120.0) / 480.0, 0.0, 1.0) * 25.0);
        } else {
            v = -25.0 - (clamp((d - 1200.0) / 1200.0, 0.0, 1.0) * 25.0);
        }
    } else if mat == 0u { // Grass
        v = 40.0 - (depth_fade * 40.0);
    } else if mat == 1u { // Road
        v = -128.0 + (depth_fade * 96.0);
    } else { // Hole (3) → 90..60
        v = 90.0 - (clamp(d / 50.0, 0.0, 1.0) * 30.0);
    }
    return i32(clamp(v, -128.0, 127.0));
}

// ─── Kernel principal ──────────────────────────────────────────────────────────
//
// 1 workgroup por ser. Cada thread processa um subconjunto de canais.
// Com SENSORY_WAVE_SIZE=223 e workgroup_size=64, usamos IDs 0..55 para palavras,
// mas aqui cada thread processa um grupo funcional de canais.
//
// Design: dividimos o trabalho por BLOCO DE CANAIS. Cada thread (id 0..63) processa
// um bloco bem definido. Não há dependência entre threads dentro do workgroup.

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)       wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let being = wgid.x;
    if being >= params.n_beings { return; }

    let tid = lid.x;
    let p = poses[being];
    let pos_x = p.pos_x;
    let pos_z = p.pos_z;
    let fwd_x = p.fwd_x;
    let fwd_z = p.fwd_z;
    let rgt_x = p.rgt_x;
    let rgt_z = p.rgt_z;
    let phase = p.phase;

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO A: ch 0-7 — FAR pool (rows 0-4, 8 bins, max-pool sobre legado)
    //
    // Motor.rs original: cols_per_fov_bin = 126/8 = 15 cols/bin (última bin pega até RETINA_COLS).
    // max-pool sobre codificação legada: Wall domina (+127 perto) → Road desaparece.
    //
    // GPU: reconstruímos legado pixel por pixel e fazemos max-pool.
    // MUDANÇA vs CPU física pura: equivalente ao motor.rs (mesma codificação legada).
    // Para migrar para física (distância mínima por bin), trocar legacy_encoding()
    // por min de dist_m e verificar impacto na evolução — deixado para decisão D.
    // ─────────────────────────────────────────────────────────────────────────
    if tid < 8u {
        let bin = tid;
        let cols_per_bin: u32 = RETINA_COLS / 8u; // 15 cols
        let lo = bin * cols_per_bin;
        let hi: u32 = select(lo + cols_per_bin, RETINA_COLS, bin == 7u);
        var fov_max: i32 = -128;
        for (var row = 0u; row < 5u; row++) {
            for (var col = lo; col < hi; col++) {
                let pk = read_retina_pixel(being, row, col);
                let v = legacy_encoding(pixel_dist_m(pk), pixel_mat(pk));
                if v > fov_max { fov_max = v; }
            }
        }
        write_channel(being, bin, fov_max);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO B: ch 8-15 — NEAR pool (rows 10-13, 8 bins, max-pool sobre legado)
    //
    // Mesma lógica que FAR, mas rows 10..14 (RETINA_ROWS/2+3..14 ≈ 20m ahead).
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 8u && tid < 16u {
        let bin = tid - 8u;
        let cols_per_bin: u32 = RETINA_COLS / 8u;
        let lo = bin * cols_per_bin;
        let hi: u32 = select(lo + cols_per_bin, RETINA_COLS, bin == 7u);
        var per_max: i32 = -128;
        for (var row = 10u; row < RETINA_ROWS; row++) {
            for (var col = lo; col < hi; col++) {
                let pk = read_retina_pixel(being, row, col);
                let v = legacy_encoding(pixel_dist_m(pk), pixel_mat(pk));
                if v > per_max { per_max = v; }
            }
        }
        write_channel(being, 8u + bin, per_max);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO C: ch 16-23 — wall_distance_fov (8 ângulos YAW, range 60m)
    //
    // Varre ±50° (MOTOR: fov_half_rad = 50°). Para cada ângulo YAW, amostra a grade
    // em 8 distâncias fixas (WALL_SAMPLES) e codifica a distância mínima até parede.
    // Equivalente EXATO ao motor.rs — mesma grade, mesma função round(), mesmos samples.
    // ATENÇÃO: motor.rs sobrescreve ch 21-27 com motion per-row DEPOIS deste bloco.
    // O kernel GPU retorna os valores deste bloco — a CPU vai sobrescrever 21-27 com motion.
    // Resultado correto: este kernel escreve 16-23; CPU sobrescreve 21-27 = ch16-20 e 21-27
    // ficam sobrepostos. A CPU DEVE escrever 21-27 por cima do que este kernel colocou.
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 16u && tid < 24u {
        let ai = tid - 16u;
        // yaw_dir: igual ao motor.rs.
        let frac = (f32(ai) - 3.5) / 3.5;
        let yaw = frac * FOV_HALF_RAD;
        let cy = cos(yaw); let sy = sin(yaw);
        let dir_x = fwd_x * cy + rgt_x * sy;
        let dir_z = fwd_z * cy + rgt_z * sy;
        let len = sqrt(dir_x * dir_x + dir_z * dir_z);
        let ndx = select(dir_x / len, 0.0, len < 1e-6);
        let ndz = select(dir_z / len, 0.0, len < 1e-6);

        // WALL_SAMPLES do motor.rs: [3, 8, 15, 25, 35, 45, 52, 60]
        let wall_samples = array<f32, 8>(3.0, 8.0, 15.0, 25.0, 35.0, 45.0, 52.0, 60.0);
        let WALL_MAX: f32 = 60.0;
        var wall_dist_min: f32 = 1e30;
        for (var si = 0u; si < 8u; si++) {
            let sd = wall_samples[si];
            let sx = pos_x + ndx * sd;
            let sz = pos_z + ndz * sd;
            let v = voxel_at(sx, sz);
            if v == 2u || v == 3u { // Wall ou Hole
                if sd < wall_dist_min { wall_dist_min = sd; }
            }
        }
        var encoded: i32;
        if wall_dist_min >= WALL_MAX {
            encoded = 0;
        } else {
            encoded = i32(clamp(127.0 * (WALL_MAX - wall_dist_min) / WALL_MAX, 1.0, 127.0));
        }
        write_channel(being, 16u + ai, encoded);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO D: ch 24-31 — wall_width_fov (largura de obstáculo a 15m por ângulo)
    //
    // Para cada ângulo, mede largura perpendicular de obstáculo em 15m à frente.
    // Offsets perpendiculares: [-3, -2, -1, +1, +2, +3] m. Equivalente exato.
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 24u && tid < 32u {
        let ai = tid - 24u;
        let frac = (f32(ai) - 3.5) / 3.5;
        let yaw = frac * FOV_HALF_RAD;
        let cy = cos(yaw); let sy = sin(yaw);
        let dir_x = fwd_x * cy + rgt_x * sy;
        let dir_z = fwd_z * cy + rgt_z * sy;
        let len = sqrt(dir_x * dir_x + dir_z * dir_z);
        let ndx = select(dir_x / len, 0.0, len < 1e-6);
        let ndz = select(dir_z / len, 0.0, len < 1e-6);
        // perp = (-ndz, ndx) — rotação 90° no plano XZ
        let perp_x = -ndz;
        let perp_z = ndx;
        let offsets = array<f32, 6>(-3.0, -2.0, -1.0, 1.0, 2.0, 3.0);
        var width: f32 = 0.0;
        for (var oi = 0u; oi < 6u; oi++) {
            let off = offsets[oi];
            let px = pos_x + ndx * 15.0 + perp_x * off;
            let pz = pos_z + ndz * 15.0 + perp_z * off;
            let v = voxel_at(px, pz);
            if v == 2u || v == 3u { width += abs(off); }
        }
        let encoded = i32(clamp(width / 6.0 * 127.0, 0.0, 127.0));
        write_channel(being, 24u + ai, encoded);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO E: ch 32-39 — wall_curvature_fov (cross product flow vs forward)
    //
    // Para cada ângulo, amostra o flow field a dist_ahead = 10 + angle_idx*1.5 m.
    // cross = forward × flow = fwd_x * flow.y − fwd_z * flow.x
    // Equivalente exato. Motor.rs: `cross * 127.0`.
    // ATENÇÃO: motor.rs sobrescreve ch 32 com c180 (curvatura 180m lookahead) DEPOIS.
    // → ch 32 da GPU será sobrescrito pela CPU. Apenas ch 33-39 persistem daqui.
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 32u && tid < 40u {
        let ai = tid - 32u;
        let frac = (f32(ai) - 3.5) / 3.5;
        let yaw = frac * FOV_HALF_RAD;
        let cy = cos(yaw); let sy = sin(yaw);
        let dir_x = fwd_x * cy + rgt_x * sy;
        let dir_z = fwd_z * cy + rgt_z * sy;
        let len = sqrt(dir_x * dir_x + dir_z * dir_z);
        let ndx = select(dir_x / len, 0.0, len < 1e-6);
        let ndz = select(dir_z / len, 0.0, len < 1e-6);
        let dist_ahead = 10.0 + f32(ai) * 1.5;
        let sx = pos_x + ndx * dist_ahead;
        let sz = pos_z + ndz * dist_ahead;
        let fl = flow_at(sx, sz);
        let cross = fwd_x * fl.y - fwd_z * fl.x;
        let encoded = i32(clamp(cross * 127.0, -127.0, 127.0));
        write_channel(being, 32u + ai, encoded);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO F: ch 40-47 — road_ahead NEAR (5..40m, a cada 5m)
    //
    // Nota: motor.rs escreve open_l em ch 36-39 e open_r em ch 40-43 ANTES, mas
    // depois o BLOCO F (road_ahead NEAR) sobrescreve ch 40-47. A semântica final
    // de ch 40-47 é road_ahead. open_l/open_r ficam em ch 36-39 e 40-43 mas são
    // sobrescritos — abaixo tratamos open_l/open_r na CPU ou ignoramos (a CPU faz).
    // A GPU escreve road_ahead em ch 40-47 que é a ÚLTIMA escritura no motor.rs.
    // Equivalente exato: get_voxel_at + get_flow_at.
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 40u && tid < 48u {
        let di = tid - 40u;
        let look_dist = 5.0 + f32(di) * 5.0; // 5, 10, ..., 40m
        let probe_x = pos_x + fwd_x * look_dist;
        let probe_z = pos_z + fwd_z * look_dist;
        let vox = voxel_at(probe_x, probe_z);
        let fl = flow_at(probe_x, probe_z);
        var road_quality: f32 = 0.0; // default
        if vox == 1u      { road_quality =  60.0; } // Road
        else if vox == 0u { road_quality = -60.0; } // Grass
        else if vox == 2u { road_quality = 127.0; } // Wall
        else              { road_quality = -127.0; } // Hole
        let flow_mag = sqrt(fl.x * fl.x + fl.y * fl.y);
        let encoded = i32(clamp(road_quality * (0.5 + flow_mag * 0.5), -127.0, 127.0));
        write_channel(being, 40u + di, encoded);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO G: ch 48-55 — road_ahead FAR (45..150m, a cada 15m)
    //
    // Usa get_height_at (bilinear na CPU, nearest aqui) + get_surface_normal (derivada
    // central 3-ponto) + get_voxel_at. Diferença de nearest vs bilinear em alturas:
    // ≤ cell_size × slope, desprezível em pistas planas. Em rampa de 10%: ≤0.3m.
    // ATENÇÃO: motor.rs sobrescreve ch 52-59 com depth_normalized DEPOIS.
    // ch 48-51 persistem daqui; ch 52-55 são sobrescritos pela CPU (depth per-row).
    // → apenas ch 48-51 têm semântica final desta GPU. ch 52-55 → CPU.
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 48u && tid < 56u {
        let si = tid - 48u;
        let horizon_dist = 45.0 + f32(si) * 15.0;
        let apex_x = pos_x + fwd_x * horizon_dist;
        let apex_z = pos_z + fwd_z * horizon_dist;
        let h_now   = height_at(apex_x, apex_z);
        let h_ahead = height_at(apex_x + fwd_x * 10.0, apex_z + fwd_z * 10.0);
        let h_delta = h_ahead - h_now;
        // Replica get_surface_normal() da CPU: derivada cruzada com step = cell_size.
        // dz = (0, hz-hc, cs); dx = (cs, hx-hc, 0)
        // normal = dz × dx = (-(cs*(hx-hc)), cs², -(hz-hc)*cs)
        // banking = atan2(normal.x, normal.y) ≈ -(hx-hc)/cs para ângulos pequenos.
        let cs = params.cell_size;
        let hc = h_now;
        let hx = height_at(apex_x + cs, apex_z);
        // Normal da superfície: dz × dx = (-(cs*(hx-hc)), cs², ...).
        // Banking = atan2(normal.x, normal.y). normal.z não entra no banking.
        // Portanto, hz (altura no eixo z+cs) não é necessário para banking.
        // Divergência vs CPU: CPU calcula normal 3D completo e normaliza. Aqui usamos
        // atan2 do numerador/denominador não-normalizado — matematicamente equivalente
        // (atan2 é invariante à escala positiva).
        let nx = -(cs * (hx - hc)); // normal.x não normalizado
        let ny = cs * cs;           // normal.y não normalizado (sempre positivo)
        let banking = atan2(nx, ny);
        let vox_far = voxel_at(apex_x, apex_z);
        var road_presence: f32 = 0.0; // default
        if vox_far == 1u      { road_presence =  1.0; } // Road
        else if vox_far == 0u { road_presence =  0.3; } // Grass
        else if vox_far == 2u { road_presence = -1.0; } // Wall
        else                  { road_presence = -0.5; } // Hole
        let apex_signal = clamp(h_delta * 50.0 + banking * 40.0 + road_presence * 30.0, -127.0, 127.0);
        write_channel(being, 48u + si, i32(apex_signal));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO H: ch 56-59 — material atual + material lookahead (3 distâncias)
    //
    // ANTES: 4× o mesmo material_code (duplicata exata, variância zero, inútil).
    // DEPOIS (2026-08-04): cada canal amostra uma distância diferente à frente.
    //   ch 56: material na posição atual         (Road=0, Grass=64, Wall=100, Hole=-127)
    //   ch 57: material a 3m à frente            — transição iminente de superfície
    //   ch 58: material a 8m à frente            — horizonte médio
    //   ch 59: material a 20m à frente           — horizonte longo
    //
    // WGSL não suporta closures: lógica de codificação expandida inline.
    // Impacto: invalida DNA treinado (semântica de ch 57-59 mudou).
    // ─────────────────────────────────────────────────────────────────────────
    if tid == 56u {
        // Codificação de material: Grass=64, Wall=100, Hole=-127, Road=0.
        // Mesma função que motor.rs material_encode().
        var mc56: i32 = 0;
        let v56 = voxel_at(pos_x, pos_z);
        if v56 == 0u { mc56 = 64; } else if v56 == 2u { mc56 = 100; } else if v56 == 3u { mc56 = -127; }
        write_channel(being, 56u, mc56);

        var mc57: i32 = 0;
        let v57 = voxel_at(pos_x + fwd_x * 3.0, pos_z + fwd_z * 3.0);
        if v57 == 0u { mc57 = 64; } else if v57 == 2u { mc57 = 100; } else if v57 == 3u { mc57 = -127; }
        write_channel(being, 57u, mc57);

        var mc58: i32 = 0;
        let v58 = voxel_at(pos_x + fwd_x * 8.0, pos_z + fwd_z * 8.0);
        if v58 == 0u { mc58 = 64; } else if v58 == 2u { mc58 = 100; } else if v58 == 3u { mc58 = -127; }
        write_channel(being, 58u, mc58);

        var mc59: i32 = 0;
        let v59 = voxel_at(pos_x + fwd_x * 20.0, pos_z + fwd_z * 20.0);
        if v59 == 0u { mc59 = 64; } else if v59 == 2u { mc59 = 100; } else if v59 == 3u { mc59 = -127; }
        write_channel(being, 59u, mc59);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO I: ch 60-63 — escape direction (run de pixels escuros no setor frontal)
    //
    // Procura o maior run contíguo de pixels com legado < -10 no setor frontal
    // (cols 32..96 = RETINA_COLS/4 até RETINA_COLS*3/4), na row RETINA_ROWS/2 - 1.
    // Centro do run → normalizado em [-1,1] relativo ao centro (31.5).
    // Motor.rs usa retina_binocular (legado). GPU reconstrói o legado.
    // ─────────────────────────────────────────────────────────────────────────
    if tid == 57u {
        let mid_row = RETINA_ROWS / 2u - 1u; // row 6 (0-indexed)
        let col_lo: u32 = RETINA_COLS / 4u;         // 31
        let col_hi: u32 = (RETINA_COLS * 3u) / 4u;  // 94
        var best_run_start: u32 = 0u;
        var best_run_len:   u32 = 0u;
        var cur_run_start:  u32 = 0u;
        var cur_run_len:    u32 = 0u;
        for (var ci = col_lo; ci < col_hi; ci++) {
            let pk = read_retina_pixel(being, mid_row, ci);
            let v = legacy_encoding(pixel_dist_m(pk), pixel_mat(pk));
            if v < -10 {
                if cur_run_len == 0u { cur_run_start = ci; }
                cur_run_len++;
                if cur_run_len > best_run_len {
                    best_run_start = cur_run_start;
                    best_run_len   = cur_run_len;
                }
            } else {
                cur_run_len = 0u;
            }
        }
        var escape_norm: f32 = 0.0;
        if best_run_len >= 2u {
            let center_of_run = f32(best_run_start) + (f32(best_run_len) - 1.0) * 0.5;
            escape_norm = clamp((center_of_run - 31.5) / 31.5, -1.0, 1.0);
        }
        let esc_i8 = i32(clamp(escape_norm * 127.0, -127.0, 127.0));
        write_channel(being, 60u, esc_i8);
        write_channel(being, 61u, esc_i8);
        write_channel(being, 62u, esc_i8);
        write_channel(being, 63u, esc_i8);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO J: ch 76-77 — distância lateral esq/dir (grid sampling, 1-40m)
    //
    // Perpendiculares ao forward: left = (-fwd_z, fwd_x), right = -left.
    // Amostra em [1,3,6,9,12,16,20,26,33,40] m (expandido de 27m → 40m).
    // Normaliza por 40.0 (máximo das amostras) para manter escala [0,127].
    // CONTA: 40m ÷ 28 m/s = 1.43 s de antecipação (frontal = 30m = 1.07 s).
    // ─────────────────────────────────────────────────────────────────────────
    if tid == 58u {
        // left_perp = (-fwd_z, fwd_x) normalizado (já é unitário pois fwd é unitário)
        let lp_x = -fwd_z;
        let lp_z =  fwd_x;
        // Sentinela 40.0 = nenhuma parede no alcance.
        var dist_l: f32 = 40.0;
        var dist_r: f32 = 40.0;
        for (var si = 0u; si < 10u; si++) {
            let sd = LAT_SAMPLES[si];
            let lx = pos_x + lp_x * sd;
            let lz = pos_z + lp_z * sd;
            let rx = pos_x - lp_x * sd;
            let rz = pos_z - lp_z * sd;
            let vl = voxel_at(lx, lz);
            let vr = voxel_at(rx, rz);
            if dist_l == 40.0 && (vl == 2u || vl == 3u) { dist_l = sd; }
            if dist_r == 40.0 && (vr == 2u || vr == 3u) { dist_r = sd; }
        }
        // 127 = encostado (sd mínimo), 0 = livre (40m ou mais).
        write_channel(being, 76u, i32(clamp((dist_l / 40.0) * 127.0, 0.0, 127.0)));
        write_channel(being, 77u, i32(clamp((dist_r / 40.0) * 127.0, 0.0, 127.0)));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO K: ch 78-80 — curvatura lookahead (40m, 70m, 120m)
    //
    // Motor.rs: curv_at(d) = atan2(cross_y, dot) onde cross_y = fwd.x*f.y - fwd.z*f.x.
    // Codificado em i8 dividindo por FRAC_PI_2. Equivalente exato.
    // ATENÇÃO: ch 78 é sobrescrito pela CPU depois (curvatura por lookahead distâncias,
    // bloco "CURVATURA ANTECIPADA" vem ANTES no motor.rs e escreve 78 com fluxo a 10m).
    // Então: a CPU sobrescreve ch 78 com curvatura a 10m APÓS este kernel.
    // O kernel escreve 78=40m, 79=70m, 80=120m — a CPU escreve 78=40m novamente
    // (fluxo forward a 10m no bloco CURVATURA, slot 78 = c40 = 40m lookahead).
    // Analisado: motor.rs linha ~1452: sensory_wave[78] = c40 (curv_at(40m)).
    // Este kernel faz o mesmo. Equivalência esperada.
    // ─────────────────────────────────────────────────────────────────────────
    if tid == 59u {
        let yn = 1.5707963267948966; // FRAC_PI_2
        let dists = array<f32, 3>(40.0, 70.0, 120.0);
        let slots = array<u32, 3>(78u, 79u, 80u);
        for (var di = 0u; di < 3u; di++) {
            let d = dists[di];
            let probe_x = pos_x + fwd_x * d;
            let probe_z = pos_z + fwd_z * d;
            let fl = flow_at(probe_x, probe_z);
            var c_signed: f32 = 0.0;
            let fmag2 = fl.x * fl.x + fl.y * fl.y;
            if fmag2 > 1e-4 {
                let cross_y = fwd_x * fl.y - fwd_z * fl.x;
                let dot = clamp(fwd_x * fl.x + fwd_z * fl.y, -1.0, 1.0);
                c_signed = atan2(cross_y, dot);
            }
            let enc = i32(clamp((c_signed / yn) * 127.0, -127.0, 127.0));
            write_channel(being, slots[di], enc);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO L: ch 82 — urgência de colisão frontal (grid forward probe)
    //
    // Amostras a [3,6,9,12,15,18,21,24,27,30] m à frente. Distância ao Wall/Hole mais próximo.
    // TTC = speed / fwd_probe — mas speed é estado do corpo (não está na GPU de percepção).
    // Sem speed, não podemos calcular TTC. Codificamos apenas fwd_probe normalizado
    // como proxy: canal = (1 - fwd_probe/30) * 127.
    // DIFERENÇA vs CPU: CPU usa ttc_inv = speed / fwd_probe. GPU usa 1 - fwd_probe/30.
    // A escala é diferente mas a informação (proximidade frontal) é equivalente.
    // A CPU PODE sobrescrever ch 82 com o valor correto (usando speed disponível).
    // Documentado: esta GPU não tem acesso a speed — não portar TTC, portar distância.
    // ─────────────────────────────────────────────────────────────────────────
    if tid == 60u {
        let fwd_samples = array<f32, 10>(3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0);
        var fwd_probe: f32 = 30.0;
        for (var si = 0u; si < 10u; si++) {
            let sd = fwd_samples[si];
            let px = pos_x + fwd_x * sd;
            let pz = pos_z + fwd_z * sd;
            let v = voxel_at(px, pz);
            if v == 2u || v == 3u {
                fwd_probe = sd;
                break;
            }
        }
        // Sem speed disponível: encodamos distância normalizada (127 = tocando, 0 = 30m+)
        let enc = i32(clamp((1.0 - fwd_probe / 30.0) * 127.0, 0.0, 127.0));
        write_channel(being, 82u, enc);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO M: ch 90-93 — dark-center tracker (sliding window)
    //
    // Encontra a janela de W=8 pixels escuros mais intensa na faixa frontal
    // (cols 16..41) da row RETINA_ROWS/2 - 1 (row 6).
    // Motor.rs usa a CODIFICAÇÃO LEGADA (v < 0 = road/escuro). GPU reconstrói.
    // Centro normalizado em [-1,1] relativo a half_col = 31.5.
    // Equivalente ao motor.rs com diferença de sub-tick na reconstrução do legado.
    // ─────────────────────────────────────────────────────────────────────────
    if tid == 61u {
        let mid_row = RETINA_ROWS / 2u - 1u; // row 6
        const W: u32 = 8u;
        // Motor.rs usa half_col = (RETINA_COLS / 2) as f32 - 0.5 = 62.5 (centro de 126 cols).
        // A janela varre 16..=(48-W)=40, mas o normalizador centra em relação a RETINA_COLS/2.
        let half_col_f: f32 = f32(RETINA_COLS / 2u) - 0.5; // 62.5

        var best_sum: f32 = -1.0;
        var best_centroid: f32 = 0.0;
        for (var start = 16u; start <= (48u - W); start++) {
            var wsum: f32 = 0.0;
            var widx: f32 = 0.0;
            for (var k = 0u; k < W; k++) {
                let pk = read_retina_pixel(being, mid_row, start + k);
                let v_leg = legacy_encoding(pixel_dist_m(pk), pixel_mat(pk));
                // d = (- v).max(0.0): pixels escuros (v negativo) têm peso > 0
                let d = f32(max(-v_leg, 0));
                wsum += d;
                widx += d * f32(start + k);
            }
            if wsum > best_sum {
                best_sum = wsum;
                if wsum > 1.0 {
                    best_centroid = widx / wsum;
                } else {
                    best_centroid = f32(start) + f32(W / 2u);
                }
            }
        }
        var dark_center_norm: f32 = 0.0;
        if best_sum > 1.0 {
            dark_center_norm = clamp((best_centroid - half_col_f) / half_col_f, -1.0, 1.0);
        }
        let dc_i8 = i32(clamp(dark_center_norm * 127.0, -127.0, 127.0));
        write_channel(being, 90u, dc_i8);
        write_channel(being, 91u, dc_i8);
        write_channel(being, 92u, dc_i8);
        write_channel(being, 93u, dc_i8);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO N: ch 94-97 — whiskers diagonais (±30°, ±60°)
    //
    // Usa racer.phase como heading absoluto + ângulo relativo.
    // Motor.rs: angle = racer.phase + angle_offset.
    // Dir = Vec3::new(angle.cos(), 0.0, angle.sin()).
    // Amostras: [1, 3, 5, 7, 10] m.
    // Equivalente exato — usa a mesma grade e mesmas amostras.
    // ─────────────────────────────────────────────────────────────────────────
    // Whiskers usam threads 62-63 — livres após BLOCO M (tid 61) e sem conflito de slots.
    if tid == 62u {
        // whiskers 0 e 1: ±30°
        let ang30 = 0.5235987756; // FRAC_PI_6
        // whisker(+30°) → ch 94
        {
            let angle = phase + ang30;
            let wx = cos(angle);
            let wz = sin(angle);
            var min_d: f32 = 10.0; // WHISKER_RANGE = 10.0
            for (var si = 0u; si < 5u; si++) {
                let sd = W_SAMPLES[si];
                let px = pos_x + wx * sd;
                let pz = pos_z + wz * sd;
                let v = voxel_at(px, pz);
                if v == 2u || v == 3u { if sd < min_d { min_d = sd; } }
            }
            let enc = i32(clamp((1.0 - min_d / 10.0) * 127.0, 0.0, 127.0));
            write_channel(being, 94u, enc);
        }
        // whisker(-30°) → ch 95
        {
            let angle = phase - ang30;
            let wx = cos(angle);
            let wz = sin(angle);
            var min_d: f32 = 10.0;
            for (var si = 0u; si < 5u; si++) {
                let sd = W_SAMPLES[si];
                let px = pos_x + wx * sd;
                let pz = pos_z + wz * sd;
                let v = voxel_at(px, pz);
                if v == 2u || v == 3u { if sd < min_d { min_d = sd; } }
            }
            let enc = i32(clamp((1.0 - min_d / 10.0) * 127.0, 0.0, 127.0));
            write_channel(being, 95u, enc);
        }
    }
    if tid == 63u {
        // whiskers 2 e 3: ±60°
        let ang60 = 1.0471975512; // FRAC_PI_3
        // whisker(+60°) → ch 96
        {
            let angle = phase + ang60;
            let wx = cos(angle);
            let wz = sin(angle);
            var min_d: f32 = 10.0;
            for (var si = 0u; si < 5u; si++) {
                let sd = W_SAMPLES[si];
                let px = pos_x + wx * sd;
                let pz = pos_z + wz * sd;
                let v = voxel_at(px, pz);
                if v == 2u || v == 3u { if sd < min_d { min_d = sd; } }
            }
            let enc = i32(clamp((1.0 - min_d / 10.0) * 127.0, 0.0, 127.0));
            write_channel(being, 96u, enc);
        }
        // whisker(-60°) → ch 97
        {
            let angle = phase - ang60;
            let wx = cos(angle);
            let wz = sin(angle);
            var min_d: f32 = 10.0;
            for (var si = 0u; si < 5u; si++) {
                let sd = W_SAMPLES[si];
                let px = pos_x + wx * sd;
                let pz = pos_z + wz * sd;
                let v = voxel_at(px, pz);
                if v == 2u || v == 3u { if sd < min_d { min_d = sd; } }
            }
            let enc = i32(clamp((1.0 - min_d / 10.0) * 127.0, 0.0, 127.0));
            write_channel(being, 97u, enc);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO O: ch 128-143 — Fóvea HD (16 bins × 8 cols, max todas as rows)
    //
    // cols_per_bin_hd = 126/16 = 7 cols/bin. Max-pool sobre legado, todas as rows.
    // Diferença vs CPU: mesma codificação legada reconstruída → equivalente.
    // ─────────────────────────────────────────────────────────────────────────
    // Threads 0-15 executam BLOCO A/B E HD Fóvea em paralelo (ifs independentes).
    // Slots distintos: ch 0-15 (BLOCOS A/B) vs ch 128-143 (HD Fóvea) — sem conflito.
    if tid < 16u {
        let bin = tid;
        let cols_per_bin: u32 = RETINA_COLS / 16u; // 7 cols/bin (126/16 = 7, última bin = 8)
        let lo = bin * cols_per_bin;
        let hi: u32 = select(lo + cols_per_bin, RETINA_COLS, bin == 15u);
        var mx: i32 = -128;
        for (var row = 0u; row < RETINA_ROWS; row++) {
            for (var col = lo; col < hi; col++) {
                let pk = read_retina_pixel(being, row, col);
                let v = legacy_encoding(pixel_dist_m(pk), pixel_mat(pk));
                if v > mx { mx = v; }
            }
        }
        write_channel(being, 128u + bin, mx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO P: ch 144-159 — Periférica HD (16 bins, max todas as rows — idêntico à fóvea)
    //
    // Motor.rs: mesmo cols_per_bin_hd, mesmas rows, mesmo max-pool. É literalmente o
    // mesmo loop que ch 128-143 (o código foi copiado no motor.rs, não é erro).
    // Usamos threads 16-31.
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 16u && tid < 32u {
        let bin = tid - 16u;
        let cols_per_bin: u32 = RETINA_COLS / 16u;
        let lo = bin * cols_per_bin;
        let hi: u32 = select(lo + cols_per_bin, RETINA_COLS, bin == 15u);
        var mx: i32 = -128;
        for (var row = 0u; row < RETINA_ROWS; row++) {
            for (var col = lo; col < hi; col++) {
                let pk = read_retina_pixel(being, row, col);
                let v = legacy_encoding(pixel_dist_m(pk), pixel_mat(pk));
                if v > mx { mx = v; }
            }
        }
        write_channel(being, 144u + bin, mx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOCO Q: ch 160-191 — Horizon scan raw (row RETINA_ROWS/2, 32 amostras)
    //
    // Motor.rs: horizon_row = RETINA_ROWS / 2 = 7; step = RETINA_COLS / 32 = 3.
    // sensory_wave[160+i] = retina_binocular[horizon_row * RETINA_COLS + i * step].
    // Usa CODIFICAÇÃO LEGADA (i8). GPU reconstrói o legado.
    // Threads 32-63 (32 threads para 32 pixels).
    // ─────────────────────────────────────────────────────────────────────────
    if tid >= 32u && tid < 64u {
        let i = tid - 32u;
        let horizon_row = RETINA_ROWS / 2u; // 7
        let step = RETINA_COLS / 32u;       // 3
        let col = i * step;
        let pk = read_retina_pixel(being, horizon_row, col);
        let v = legacy_encoding(pixel_dist_m(pk), pixel_mat(pk));
        write_channel(being, 160u + i, v);
    }

    // ── NOTA: ch 21-27 e ch 98 (motion per-row) ─────────────────────────────────
    //
    // O BLOCO C escreveu wall_dist_fov em ch 16-23 (ch 16-20 corretos, ch 21-23 errados).
    // Motor.rs sobrescrevia 21-27 com motion_per_row APÓS o BLOCO C.
    //
    // Antes: não tínhamos o frame anterior da retina → zerávamos ch 21-23 via atomicAnd
    // (0 = "sem movimento") → CHAMPION via 0 constante → navegação sem sinal de movimento.
    //
    // AGORA: perception_motion.wgsl (3º kernel) compara buf_retina_fisica (tick t) vs
    // buf_retina_prev (tick t-1) e escreve ch 21-27 + 98 via atomicOr.
    // O atomicOr do motion kernel sobrescreve o que o BLOCO C escreveu em ch 21-23.
    // (buf_output foi zerado em clear_buffer ANTES dos dispatches, então o BLOCO C
    // escreve wall_dist via atomicOr, e o motion kernel sobrescreve com o delta correto
    // via um segundo atomicOr — mas atomicOr acumula bits, não sobrescreve!)
    //
    // SOLUÇÃO: o motion kernel usa atomicAnd primeiro para limpar ch 21-23, depois
    // escreve via atomicOr. Alternativamente, mantemos o atomicAnd aqui (BLOCO C
    // nunca deveria ter escrito nesses slots — é uma inconsistência do motor.rs).
    //
    // Conclusão: o atomicAnd abaixo É NECESSÁRIO para limpar os bits que o BLOCO C
    // escreveu em ch 21-23. O motion kernel então faz atomicOr com o valor correto.
    // Sem o atomicAnd: ch 21 = (wall_dist_do_bloco_C) | (motion_do_kernel) = LIXO.
    if tid == 0u {
        // Zera ch 21, 22, 23 para que o motion kernel possa escrever via atomicOr.
        // ch 20 (byte 0) é preservado (escrito corretamente pelo BLOCO C = wall_dist a 0°).
        let base_w5 = being * WAVE_U32S + 5u;  // palavra 5 = ch 20,21,22,23
        atomicAnd(&output[base_w5], 0x000000FFu);  // preserva byte 0 (ch 20), zera 21-23
    }

    // Canais temporais (85, 86, 88, 102-110, 118-120, 123) foram movidos para
    // perception_temporal.wgsl. O segundo kernel roda em seguida no mesmo encoder
    // e escreve no mesmo buf_output (canais distintos, sem race condition).
    // O terceiro kernel (perception_motion.wgsl) escreve ch 21-27 e 98.
    // Motivo da divisão: o shader combinado (~1050 linhas) estourava o compilador
    // ACO do RADV (SIGSEGV em create_compute_pipeline), igual ao physics.wgsl
    // que também foi dividido em 3 kernels. A solução é a mesma: três dispatches
    // encadeados no mesmo CommandEncoder, sem readback entre eles.
}
