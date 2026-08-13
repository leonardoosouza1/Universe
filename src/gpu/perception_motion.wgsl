// PERCEPÇÃO — kernel de motion (fluxo óptico por linha).
//
// # O que este kernel faz
//
// Calcula os canais ch 21-27 (motion_per_row): diferença absoluta de distância
// entre o frame atual da retina (buf_retina_fisica) e o frame anterior
// (buf_retina_prev), sumada por linha.
//
// # Equivalência com motor.rs
//
// motor.rs (linhas 854-869):
//   for row in 0..RETINA_ROWS {
//     motion_sum = Σ |cur_binocular[row×126+col] - prev_cortex[row×126+col]|
//     motion_norm = min(127, (motion_sum / 8000.0) * 127.0) as i8
//     sensory_wave[21 + row] = motion_norm  // row 0..6 → ch 21-27
//     sensory_wave[98] = motion_norm         // row 7 → ch 98
//   }
//
// motor.rs usa bytes (u8 de retina_binocular). Este shader usa dist em mm (u32 >> 8).
// 1 byte ≈ 255 unidades de [0,60m] → 255/60 ≈ 4.25 unidades/m.
// Para equivalência: delta_dist_m × 4.25 ≈ delta_byte.
// Normalizador: motor usa sum_max ≈ 8000 bytes. Em metros: 8000 / 4.25 ≈ 1882 m.
// Usamos 1882.0 como normalizador (validado: 1 byte delta → 1 byte no canal).
//
// # Dados lidos
//
// binding 0 — Params (uniform): n_beings, grid_size, cell_size (não usados aqui)
// binding 1 — buf_retina_fisica: retina do tick ATUAL (N × 1764 u32, packed mm|mat)
// binding 2 — buf_retina_prev:   retina do tick ANTERIOR (mesmo formato)
// binding 3 — buf_output:        sensory_wave (N × WAVE_U32S u32, read_write, atomicOr)
//
// # Dispatch
//
// workgroup_size = (8, 1, 1) — 8 threads por ser, tid 0-6 = linhas 0-6, tid 7 = linha 7
// dispatch_workgroups(n_beings, 1, 1) — 1 workgroup por ser
// Cada thread processa 126 pixels (row completa) e escreve 1 canal.
//
// # Atomics
//
// O shader usa atomicOr para escrever bytes individuais em buf_output, igual aos outros
// kernels de percepção. Garante ausência de corrida com spatial e temporal (canais distintos).

struct Params {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    _pad:      f32,
}

@group(0) @binding(0) var<uniform>         params       : Params;
@group(0) @binding(1) var<storage, read>   retina_cur   : array<u32>;  // tick atual
@group(0) @binding(2) var<storage, read>   retina_prev  : array<u32>;  // tick anterior
@group(0) @binding(3) var<storage, read_write> output   : array<atomic<u32>>;

const RETINA_ROWS:  u32 = 14u;
const RETINA_COLS:  u32 = 126u;
const RETINA_TOTAL: u32 = 1764u;   // 14 × 126
const WAVE_U32S:    u32 = 48u;     // ceil(192/4)

// Normalizador: motor.rs usa sum_max ≈ 8000 bytes de u8.
// Em distância metro: 8000 / (255 / 60) ≈ 1882 m de soma de delta-dist.
// Empiricamente: 2000 m dá margem para deltas grandes em curvas bruscas.
const MOTION_NORM: f32 = 2000.0;

// Normalizador hemisférico (ch 83-84): motor.rs usa sum / 128, onde sum é em bytes u8.
// 63 cols × 14 rows × 255 = ~224.910 unidades máx em bytes.
// Para equivalência com "motion_left / 128" do motor.rs, onde motion_left é a soma
// de deltas em bytes unsigned:
//   motor.rs: delta_byte = |cur_byte - prev_byte|, max=255/pixel.
//   GPU: delta_m = |dist_cur - dist_prev| em metros, max ≈ 60m.
//   Fator: 255 / 60 ≈ 4.25 bytes/m.
// Então: motor.rs usa sum/128; GPU usa (sum_m × 4.25) / 128 → sum_m / 30.1.
// Usamos HEMI_NORM = 30.0 (arredondado) para ch 83-84.
const HEMI_NORM: f32 = 30.0;

// Extrai distância em metros do pixel packed (mm << 8 | mat).
fn pixel_dist_m(packed: u32) -> f32 {
    return f32(packed >> 8u) / 1000.0;
}

// Escreve um byte (valor i8) no slot correto de buf_output via atomicOr.
// Cada u32 empacota 4 bytes little-endian: byte0=ch%4==0, ..., byte3=ch%4==3.
// atomicOr: o buf_output foi zerado antes dos dispatches (enc.clear_buffer),
// portanto OR com o byte desejado produz o valor final correto.
fn write_channel(being: u32, ch: u32, val: i32) {
    let word_idx = being * WAVE_U32S + ch / 4u;
    let byte_pos = (ch % 4u) * 8u;
    // Clip para i8 ([-127, 127] → 0 = parado, 127 = máximo movimento detectado).
    let v = u32(clamp(val, 0, 127)) & 0xFFu;
    atomicOr(&output[word_idx], v << byte_pos);
}

// workgroup_size(10): 8 threads para rows 0-7 + 2 threads extras para ch83/84.
// RDNA2 tem wave32; 10 threads = 1 wave incompleto mas executado em 1 wave (32 slots).
// Não há ganho em aumentar para 16 — o trabalho é O(RETINA_COLS) = 126 por thread.
@compute @workgroup_size(10, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let being = gid.x;
    let tid   = lid.x;  // 0-9: thread dentro do workgroup

    if being >= params.n_beings { return; }

    // ── Threads 0-7: motion por row (ch 21-27, 98) ────────────────────────────
    if tid < 8u {
        let row  = tid;
        let base = being * RETINA_TOTAL + row * RETINA_COLS;

        // Soma |dist_cur - dist_prev| por coluna nesta linha.
        var delta_sum: f32 = 0.0;
        for (var col = 0u; col < RETINA_COLS; col++) {
            let idx = base + col;
            let dist_cur  = pixel_dist_m(retina_cur[idx]);
            let dist_prev = pixel_dist_m(retina_prev[idx]);
            delta_sum += abs(dist_cur - dist_prev);
        }

        // Normaliza para i8: 0 = parado, 127 = movimento máximo.
        let motion = i32(clamp((delta_sum / MOTION_NORM) * 127.0, 0.0, 127.0));

        // row 0-6 → ch 21-27; row 7 → ch 98
        if row < 7u {
            write_channel(being, 21u + row, motion);
        } else {
            write_channel(being, 98u, motion);
        }
        return;
    }

    // ── Thread 8: fluxo óptico hemisférico esquerdo (ch 83) ──────────────────
    // motor.rs L1057-1069: soma |delta| nas colunas esquerda (0..half) das rows
    // próximas ((RETINA_ROWS/2 - 1) e (RETINA_ROWS/2 + 1)) — 2 rows × 63 cols.
    // Aqui somamos TODAS as rows (mais sinal) pelas colunas 0..63 (metade esq).
    // Normalizador HEMI_NORM = 30.0 (equivalente a motor.rs sum_byte / 128).
    if tid == 8u {
        let half = RETINA_COLS / 2u;  // 63
        var sum_l: f32 = 0.0;
        // Usa rows do meio (próximas, mais relevantes para navegação lateral).
        // rows (RETINA_ROWS/2 - 1) = 6 e (RETINA_ROWS/2 + 1) = 8 — mesmas do motor.rs.
        let row_a = RETINA_ROWS / 2u - 1u;  // row 6
        let row_b = RETINA_ROWS / 2u + 1u;  // row 8
        for (var col = 0u; col < half; col++) {
            let idx_a = being * RETINA_TOTAL + row_a * RETINA_COLS + col;
            let idx_b = being * RETINA_TOTAL + row_b * RETINA_COLS + col;
            sum_l += abs(pixel_dist_m(retina_cur[idx_a]) - pixel_dist_m(retina_prev[idx_a]));
            sum_l += abs(pixel_dist_m(retina_cur[idx_b]) - pixel_dist_m(retina_prev[idx_b]));
        }
        let val = i32(clamp((sum_l / HEMI_NORM) * 127.0, 0.0, 127.0));
        write_channel(being, 83u, val);
        return;
    }

    // ── Thread 9: fluxo óptico hemisférico direito (ch 84) ───────────────────
    // Espelho do thread 8, mas colunas half..RETINA_COLS (metade direita).
    if tid == 9u {
        let half = RETINA_COLS / 2u;  // 63
        var sum_r: f32 = 0.0;
        let row_a = RETINA_ROWS / 2u - 1u;  // row 6
        let row_b = RETINA_ROWS / 2u + 1u;  // row 8
        for (var col = half; col < RETINA_COLS; col++) {
            let idx_a = being * RETINA_TOTAL + row_a * RETINA_COLS + col;
            let idx_b = being * RETINA_TOTAL + row_b * RETINA_COLS + col;
            sum_r += abs(pixel_dist_m(retina_cur[idx_a]) - pixel_dist_m(retina_prev[idx_a]));
            sum_r += abs(pixel_dist_m(retina_cur[idx_b]) - pixel_dist_m(retina_prev[idx_b]));
        }
        let val = i32(clamp((sum_r / HEMI_NORM) * 127.0, 0.0, 127.0));
        write_channel(being, 84u, val);
        return;
    }
}
