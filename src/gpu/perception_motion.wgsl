// PERCEPÇÃO — kernel de motion (fluxo óptico por linha).
//
// # O que este kernel faz
//
// Calcula os canais ch 21-27 (motion_per_row): diferença absoluta de distância
// entre o frame atual da retina (buf_retina_fisica) e o frame anterior
// (buf_retina_prev), sumada por linha.
//
// Também calcula ch 83-84 (motion hemisférico) e, desde 2026-08-14, ch 114-117
// (approach hemisférico + fóvea) — ver bloco "FASE CANAIS" abaixo.
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
// workgroup_size = (11, 1, 1) — 11 threads por ser: tid 0-6 = linhas 0-6 (ch21-27),
// tid 7 = linha 7 (ch98), tid 8/9 = hemisférico esq/dir (ch83/84 + ch114/115,
// FASE CANAIS 2026-08-14), tid 10 = fóvea (ch116/117, idem).
// dispatch_workgroups(n_beings, 1, 1) — 1 workgroup por ser
// Cada thread das linhas 0-7 processa 126 pixels (row completa) e escreve 1 canal.
//
// # Atomics
//
// Most channels use atomicOr to write individual bytes into the zeroed output buffer.
// Channels 21-27 are legacy overwrite slots: the spatial pass writes wall probes
// there first, so this pass explicitly replaces each owned byte with retinal motion.
//
// # BODY BLOCK — ownership handoff (BODY PHASE, 2026-08-17)
//
// CONTRACT WITH THE INTEGRATOR, registered decision (mission rule 7):
//
//   params.body_on == 0 (DEFAULT) -> THIS kernel writes ch98 and ch114-117, exactly
//                                    as before, bit for bit.
//   params.body_on == 1           -> this kernel writes NONE of the five;
//                                    perception_temporal.wgsl writes all five from
//                                    the BODY BLOCK (buf_temporal[58-65]).
//
// ch21-27 (per-row motion) and ch83-84 (hemispheric flow) are NOT part of the
// handoff — they keep coming out of here in both states. Only the five channels
// MEASURED constant for a whole run change owner (`piloto_forja.rs:49`: "`sd` has 6
// degenerate channels (98, 114, 115, 116, 117, 124) with `sd = 1e-12`"; the sixth,
// ch124, was never this kernel's).
//
// Why the guard is on BOTH sides and not just one: buf_output is written with
// atomicOr over a zeroed buffer. Two kernels writing the same byte raise no error
// and no panic — they produce the OR of the two values, a plausible number. Neither
// side writing raises no error either: it produces a mute channel. Both failures are
// silent, which is why both sides read THE SAME field of THE SAME uniform, built
// once per tick from ONE static (gpu::perception::BODY_CHANNELS_ON). There is no
// state in which the two can disagree.
//
// Detection, if they ever do disagree: `cargo test --release --test wgsl_validation`
// (parse/type-check) does not see semantics; what does is `--auditar-visao`
// (per-index variance: a mute channel shows sigma=0, a double write shows a value
// outside the registered range) and the parity test in `perception_body_test.rs`.

// Uniform shared by the THREE perception pipelines (spatial/temporal/motion) — the
// same `buf_params`. The layout mirrors `ParamsGpu` in perception.rs (32 bytes, a
// multiple of 16 because WGSL aligns a uniform struct that way). Change it here and
// you change it in all three .wgsl AND on the host, in the same commit.
struct Params {
    grid_size:  u32,
    n_beings:   u32,
    cell_size:  f32,
    perif_real: f32,  // unused in this kernel; present so the layout matches
    // BODY BLOCK (BODY PHASE, 2026-08-17): 1.0 = body-block channels on. Comes from
    // ONE single static (gpu::perception::BODY_CHANNELS_ON), read once per tick on
    // the host — which is why temporal.wgsl and motion.wgsl can NEVER disagree about
    // who writes ch98/ch114-117.
    body_on:    f32,
    _pad0:      f32,
    _pad1:      f32,
    _pad2:      f32,
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
//
// APENAS para canais NÃO-NEGATIVOS (motion magnitude, sempre >=0) — por isso o
// clamp inferior é 0, não -128. Canais com sinal (approach/fóvea abaixo) usam
// `write_channel_signed`.
fn write_channel(being: u32, ch: u32, val: i32) {
    let word_idx = being * WAVE_U32S + ch / 4u;
    let byte_pos = (ch % 4u) * 8u;
    // Clip para i8 ([-127, 127] → 0 = parado, 127 = máximo movimento detectado).
    let v = u32(clamp(val, 0, 127)) & 0xFFu;
    atomicOr(&output[word_idx], v << byte_pos);
}

// Replaces one packed byte while preserving the other three channels in its u32.
// Rows 0-6 run concurrently and share packed words, so both operations must be atomic.
fn overwrite_channel(being: u32, ch: u32, val: i32) {
    let word_idx = being * WAVE_U32S + ch / 4u;
    let byte_pos = (ch % 4u) * 8u;
    let byte_mask = 0xFFu << byte_pos;
    atomicAnd(&output[word_idx], ~byte_mask);
    let v = u32(clamp(val, 0, 127)) & 0xFFu;
    atomicOr(&output[word_idx], v << byte_pos);
}

// Idêntica a `write_channel`, mas preserva sinal (clamp em [-128,127] como
// perception.wgsl/perception_temporal.wgsl fazem) — necessária para ch 114-116
// (approach/fóvea COM SINAL: positivo = parede se aproximando).
fn write_channel_signed(being: u32, ch: u32, val: i32) {
    let word_idx = being * WAVE_U32S + ch / 4u;
    let byte_pos = (ch % 4u) * 8u;
    let clamped = clamp(val, -128, 127);
    let v = u32(clamped & 0xFF) & 0xFFu;
    atomicOr(&output[word_idx], v << byte_pos);
}

// ── FASE CANAIS (2026-08-14) — ch 114-117, antes só alimentados por motor.rs ────
//
// DIAGNÓSTICO (docs/PLANO_EVOLUCAO.md + auditoria 2026-08-14): ch101 e ch114-117
// eram escritos SÓ no caminho Bevy CPU (motor.rs) — `gpu_evolve` (o binário de
// treino/evolução) nunca importa `motor.rs`, então nesse caminho os canais ficavam
// sem fonte nenhuma (reprovados por "sem_fonte" OU "sem variância", dependendo de
// como o buffer zerado é lido). ch101 foi resolvido em `perception_temporal.wgsl`
// (reusa `curv_at_d`, já validado para ch111-113). ch114-117 precisam de retina
// atual × anterior — dados que ESTE kernel já tem (retina_cur/retina_prev), então
// a fonte real (não uma reimplementação) é aqui.
//
// Fórmula original (motor.rs:1187-1226, CODIFICAÇÃO LEGADA em bytes, "positivo =
// byte sobe = parede ficando mais perto"):
//   approach_left/right (ch114/115): soma SIGNED de `cur_byte - prev_byte` na row
//     "posterior" (RETINA_ROWS/2 + 1 = row 8), colunas 0..half / half..RETINA_COLS.
//   fovea_approach/motion (ch116/117): soma signed/abs de `cur_byte - prev_byte`
//     na row "frontal" (RETINA_ROWS/2 - 1 = row 6), colunas 24..40 (~±15° central).
//
// Adaptação para FÍSICA (metros, não byte legado — mesma migração já feita para
// ch21-27/83-84 neste arquivo): em metros, MENOR distância = mais perto, o
// contrário do byte legado. "Aproximando" vira `dist_prev_m - dist_cur_m > 0`
// (distância caindo). Reusamos as MESMAS linhas (6 e 8) e a MESMA metade de
// colunas que ch83/84 já leem — evita uma 3ª leitura de retina e mantém a
// normalização (HEMI_NORM) comparável, já que é a mesma amostra, só sem abs().
// Fóvea usa só row 6 (a linha "frontal" do motor.rs), colunas 24..40 — 16
// amostras. FOVEA_NORM foi escalado a partir de HEMI_NORM proporcionalmente ao
// número de amostras (16 vs 126 de HEMI): 30.0 * 16/126 ≈ 3.8 → arredondado a 4.0.
// NÃO validado por medição (GPU ocupada com baterias em 2026-08-14) — validar com
// `--auditar-visao` (σ_temporal/σ_entre destes 4 canais) antes de fechar a fase.
const FOVEA_NORM: f32 = 4.0;

// workgroup_size(11): 8 threads para rows 0-7 + 3 threads extras (ch83/84 hemisférico
// + ch114-117 approach/fóvea, fase canais 2026-08-14).
// RDNA2 tem wave32; 11 threads = 1 wave incompleto mas executado em 1 wave (32 slots).
// Não há ganho em aumentar para 16 — o trabalho é O(RETINA_COLS) = 126 por thread.
@compute @workgroup_size(11, 1, 1)
fn main(@builtin(workgroup_id)        workgroup: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let being = workgroup.x;
    let tid   = lid.x;  // 0-10: thread dentro do workgroup (11 threads, FASE CANAIS 2026-08-14)

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
        // BODY BLOCK: with params.body_on, ch98 becomes PredatorBearing and is
        // written by perception_temporal.wgsl — this kernel GOES SILENT. See the
        // "BODY BLOCK — ownership handoff" block at the top of this file.
        if row < 7u {
            overwrite_channel(being, 21u + row, motion);
        } else if params.body_on <= 0.5 {
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
        var signed_l: f32 = 0.0; // FASE CANAIS 2026-08-14: ch114 (approach esquerda)
        // Usa rows do meio (próximas, mais relevantes para navegação lateral).
        // rows (RETINA_ROWS/2 - 1) = 6 e (RETINA_ROWS/2 + 1) = 8 — mesmas do motor.rs.
        let row_a = RETINA_ROWS / 2u - 1u;  // row 6
        let row_b = RETINA_ROWS / 2u + 1u;  // row 8
        for (var col = 0u; col < half; col++) {
            let idx_a = being * RETINA_TOTAL + row_a * RETINA_COLS + col;
            let idx_b = being * RETINA_TOTAL + row_b * RETINA_COLS + col;
            let d_a = pixel_dist_m(retina_prev[idx_a]) - pixel_dist_m(retina_cur[idx_a]);
            let d_b = pixel_dist_m(retina_prev[idx_b]) - pixel_dist_m(retina_cur[idx_b]);
            sum_l += abs(d_a) + abs(d_b);
            signed_l += d_a + d_b; // positivo = distância caindo = parede se aproximando
        }
        let val = i32(clamp((sum_l / HEMI_NORM) * 127.0, 0.0, 127.0));
        write_channel(being, 83u, val);
        // ch114: approach hemisférico esquerdo, COM SINAL — ver bloco "FASE CANAIS" acima.
        // BODY BLOCK: with body_on, ch114 becomes StationBearing (temporal.wgsl) and
        // this kernel goes silent — ch83 above is still written in both states.
        if params.body_on <= 0.5 {
            let approach_l = i32(clamp((signed_l / HEMI_NORM) * 127.0, -127.0, 127.0));
            write_channel_signed(being, 114u, approach_l);
        }
        return;
    }

    // ── Thread 9: fluxo óptico hemisférico direito (ch 84) ───────────────────
    // Espelho do thread 8, mas colunas half..RETINA_COLS (metade direita).
    if tid == 9u {
        let half = RETINA_COLS / 2u;  // 63
        var sum_r: f32 = 0.0;
        var signed_r: f32 = 0.0; // FASE CANAIS 2026-08-14: ch115 (approach direita)
        let row_a = RETINA_ROWS / 2u - 1u;  // row 6
        let row_b = RETINA_ROWS / 2u + 1u;  // row 8
        for (var col = half; col < RETINA_COLS; col++) {
            let idx_a = being * RETINA_TOTAL + row_a * RETINA_COLS + col;
            let idx_b = being * RETINA_TOTAL + row_b * RETINA_COLS + col;
            let d_a = pixel_dist_m(retina_prev[idx_a]) - pixel_dist_m(retina_cur[idx_a]);
            let d_b = pixel_dist_m(retina_prev[idx_b]) - pixel_dist_m(retina_cur[idx_b]);
            sum_r += abs(d_a) + abs(d_b);
            signed_r += d_a + d_b;
        }
        let val = i32(clamp((sum_r / HEMI_NORM) * 127.0, 0.0, 127.0));
        write_channel(being, 84u, val);
        // ch115: approach hemisférico direito, COM SINAL.
        // BODY BLOCK: with body_on, ch115 becomes StationProx (temporal.wgsl) — silent here.
        if params.body_on <= 0.5 {
            let approach_r = i32(clamp((signed_r / HEMI_NORM) * 127.0, -127.0, 127.0));
            write_channel_signed(being, 115u, approach_r);
        }
        return;
    }

    // ── Thread 10: fóvea — approach + motion absoluto (ch 116, 117) ──────────
    // FASE CANAIS 2026-08-14. Row 6 (RETINA_ROWS/2 - 1, a mesma "frontal" do
    // motor.rs), colunas 24..40 (~±15° centrais, a fóvea) — ver bloco "FASE
    // CANAIS" acima para a derivação de FOVEA_NORM e a adaptação de sinal.
    // BODY BLOCK: with body_on, ch116/117 become StationCharge/OwnFuel
    // (temporal.wgsl) and this whole thread has nothing left to write — the guard
    // sits at the top, which also skips the 16-column loop.
    if tid == 10u && params.body_on <= 0.5 {
        let row = RETINA_ROWS / 2u - 1u; // row 6
        var signed_sum: f32 = 0.0;
        var abs_sum: f32 = 0.0;
        for (var col = 24u; col < 40u; col++) {
            let idx = being * RETINA_TOTAL + row * RETINA_COLS + col;
            let d = pixel_dist_m(retina_prev[idx]) - pixel_dist_m(retina_cur[idx]);
            signed_sum += d;
            abs_sum += abs(d);
        }
        // ch116: fovea_approach, COM SINAL (positivo = parede vindo de frente).
        let approach = i32(clamp((signed_sum / FOVEA_NORM) * 127.0, -127.0, 127.0));
        write_channel_signed(being, 116u, approach);
        // ch117: fovea_motion_abs, magnitude (qualquer movimento na fóvea).
        let motion = i32(clamp((abs_sum / FOVEA_NORM) * 127.0, 0.0, 127.0));
        write_channel(being, 117u, motion);
        return;
    }
}
