// RETINA DIGEST — CfC de percepção + min_wall_ahead + edge_row6/7, tudo na GPU.
//
// # Por que este kernel existe
//
// `context.rs::tick()` (gpu_evolve) fazia, TODO tick, um readback de 340 KB (48 seres ×
// 1764 px × 4 B) da retina inteira e processava 84.672 pixels em laço Rust só para
// produzir, por ser: 16 floats de estado CfC (ch192-207), `min_wall_ahead` (dreno de
// vida por parede) e `edge_row6`/`edge_row7` (ch99/ch100). Esse readback+laço era o
// gargalo medido do tick (2.30 ms contra 0.63 ms do tick GPU puro).
//
// Este kernel roda DEPOIS de `retina.wgsl`, no MESMO `CommandEncoder`, consumindo o
// buffer de retina física que já está em VRAM — sem NENHUM round-trip à CPU no laço
// quente. Só desce à CPU o resultado pequeno: 3 floats por ser (`extra_out`).
//
// # O que este kernel NÃO faz
//
// O laço de vida (wall_ticks, dreno de vida, estagnação, combustível, -1 tick de
// life_left) CONTINUA na CPU — ele só passa a consumir `min_wall_ahead` vindo daqui em
// vez de recalculá-lo dos pixels. Ver `context.rs::tick()`, bloco "Atualiza CfC
// percepção" (agora reduzido a: ler `extra_out`, aplicar os drenos).
//
// # A equação do CfC — UMA fonte, dois lugares que a espelham
//
// A equação em si mora em `iara_cfc::CfcCell::step` (crate externa, compartilhada por
// todo o ecossistema IARA) — ver `src/math/cfc_perception.rs`, que é o front door deste
// crate para ela e documenta a fórmula por extenso. ESTE shader é o espelho GPU; se a
// equação mudar, as DUAS cópias (lá e aqui) têm de mudar juntas — não existe uma
// terceira cópia em lugar nenhum (Leonardo: "sem CfC espalhado").
//
// matvec aqui usa um `acc` sequencial (`for c { acc += w[r,c]*x[c] }`), replicando o
// loop de `iara_cfc::matvec` — NÃO usar `dot()` do WGSL, que permite ao compilador
// reassociar a soma e divergir em ULPs do resultado calculado na CPU.
//
// # Pipeline até virar x[48] (entrada do CfC) — espelha `perception/retina.rs`
//
// 1. `legacy_encoding(dist_m, material)` por pixel — CÓPIA LITERAL de
//    `perception.wgsl:251` (WGSL não importa função entre arquivos neste projeto;
//    qualquer correção tem de tocar as duas cópias, mais `gpu::legacy_encoding` em
//    Rust — é a mesma dívida documentada em `perception.wgsl` para uma 4ª cópia).
// 2. Col max-pool: 126 colunas → 16 bins por linha (`cols_per_bin = 126/16 = 7`, o
//    último bin absorve o resto até 126) — espelha `RetinaSensor::sense_retina`.
//    O máximo é tirado sobre o valor ASSINADO (i32, mesma faixa de um i8), só DEPOIS
//    deslocado para [0,255] via `+128` (equivalente a `(max_val as i16 + 128) as u8`).
// 3. Zone-pool: far=linhas[0,5) MEAN, mid=linhas[5,10) MEAN, near=linhas[10,14) MAX —
//    espelha `retina_zones()`. Cada valor pooled (0..255) vira f32 via `/127.5 - 1.0`
//    DURANTE a agregação da zona (não antes) — espelha `ZoneConfig::pool`.
// 4. x = [far(16) | mid(16) | near(16)] = 48 floats — mesma ordem de `pool_flat()`.
//
// # ch192-207 — a MESMA defasagem de 1 tick que já existia
//
// A CPU antiga injetava `hidden[]` em `buf_temporal[33-48]` ANTES do dispatch (estado
// do CfC calculado no tick ANTERIOR, via readback pós-submit) e só `perception_temporal
// .wgsl` convertia esse hidden em canal. Ou seja: ch192-207 no tick t sempre refletia o
// CfC até o tick t-1. Este kernel replica a defasagem DENTRO do próprio dispatch:
//   1) escreve ch192-207 usando `cfc_state` COMO ESTAVA no início do tick (= h após o
//      tick t-1);
//   2) só DEPOIS processa a retina deste tick e escreve o h NOVO de volta em
//      `cfc_state`, para o PRÓXIMO tick ler.
// A fórmula de conversão hidden→canal é a aproximação x/(1+|x|) que já estava em
// `perception_temporal.wgsl` (NÃO o `tanh()` nativo do WGSL) — copiada literalmente
// abaixo para não mudar o valor que o cérebro recebe.
//
// # Bindings
//
// binding 0 (uniform)          params: n_beings
// binding 1 (storage, read)    retina: array<u32>       — N × 1764, packed (mm<<8|mat)
// binding 2 (storage, read)    cfc_weights: array<f32>   — 1312 floats, buffer ÚNICO
//   compartilhado por todos os seres (pesos fixos e idênticos — ver achado "pesos do
//   CfC nunca carregados do genoma" em src/math/cfc_perception.rs). Layout:
//   [w_xi(768) | w_hi(256) | w_f(256) | b_i(16) | b_f(16)].
// binding 3 (storage, read_write) cfc_state: array<f32>  — N × 16, RESIDENTE na VRAM
//   entre ticks. Zerado por `reset()` (host) entre gerações.
// binding 4 (storage, read_write) output: array<atomic<u32>> — MESMO `buf_output` da
//   percepção (perception.rs) — este kernel só escreve ch192-207 aqui, via atomicOr.
//   PRÉ-CONDIÇÃO: já foi zerado por `enc.clear_buffer()` (percept.bind_and_dispatch o
//   faz antes de qualquer dispatch) e este kernel deve rodar DEPOIS desse clear.
// binding 5 (storage, read_write) extra_out: array<f32>  — N × 3:
//   [0] min_wall_ahead (m, sentinela grande se não há parede à frente nas colunas
//       centrais), [1] edge_row6 normalizado [0,1], [2] edge_row7 normalizado [0,1].
//
// # Dispatch
//
// `@workgroup_size(1)`, `dispatch_workgroups(n_beings, 1, 1)` — 1 thread por ser,
// espelhando `perception_temporal.wgsl` (padrão comprovado seguro contra o SIGSEGV do
// compilador ACO/RADV que loops sobre storage buffers com workgroup_size>1 provocavam).

struct Params {
    n_beings: u32,
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
}

@group(0) @binding(0) var<uniform>             params:      Params;
@group(0) @binding(1) var<storage, read>       retina:      array<u32>;
@group(0) @binding(2) var<storage, read>       cfc_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> cfc_state:   array<f32>;
@group(0) @binding(4) var<storage, read_write> output:      array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> extra_out:   array<f32>;

const RETINA_ROWS:  u32 = 14u;
const RETINA_COLS:  u32 = 126u;
const RETINA_TOTAL: u32 = 1764u;
const WAVE_U32S:    u32 = 48u;

const POOL_BINS:    u32 = 16u;   // PERCEPTION_BINS
const CFC_HIDDEN:   u32 = 16u;   // CFC_HIDDEN (math/cfc_perception.rs)
const CFC_INPUT:    u32 = 48u;   // CFC_INPUT  (3 zonas × 16 bins)

// Offsets dentro de `cfc_weights` — espelha `math::cfc_perception::flat_weights()`.
const OFF_W_XI: u32 = 0u;                                   // 16×48 = 768
const OFF_W_HI: u32 = OFF_W_XI + CFC_HIDDEN * CFC_INPUT;     // 768
const OFF_W_F:  u32 = OFF_W_HI + CFC_HIDDEN * CFC_HIDDEN;    // 1024
const OFF_B_I:  u32 = OFF_W_F  + CFC_HIDDEN * CFC_HIDDEN;    // 1280
const OFF_B_F:  u32 = OFF_B_I  + CFC_HIDDEN;                 // 1296

const VOXEL_WALL:    u32 = 2u;
// Colunas centrais (terço do meio dos 126 pixels) — espelha context.rs:823-824.
const WALL_COL_LO:   u32 = RETINA_COLS / 3u;       // 42
const WALL_COL_HI:   u32 = 2u * RETINA_COLS / 3u;  // 84
// Sentinela "sem parede à frente" — só precisa ser >= WALL_NEAR_M (3.0 m, em
// statics.rs) para o consumidor da CPU (`if min_wall_ahead < WALL_NEAR_M`) se
// comportar como o `f32::MAX` que o Rust usava. Não precisa ser o MAX exato.
const NO_WALL_SENTINEL: f32 = 1.0e9;

// ─── legacy_encoding — CÓPIA LITERAL de perception.wgsl:251 ───────────────────────
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
    } else { // Hole (3)
        v = 90.0 - (clamp(d / 50.0, 0.0, 1.0) * 30.0);
    }
    return i32(clamp(v, -128.0, 127.0));
}

// ─── write_channel — CÓPIA LITERAL de perception.wgsl:208 ─────────────────────────
fn write_channel(being: u32, ch: u32, val: i32) {
    let clamped  = clamp(val, -128, 127);
    let byte_val = u32(clamped & 0xFF);
    let base     = being * WAVE_U32S + ch / 4u;
    atomicOr(&output[base], byte_val << ((ch % 4u) * 8u));
}

fn read_retina_pixel(being: u32, row: u32, col: u32) -> u32 {
    return retina[being * RETINA_TOTAL + row * RETINA_COLS + col];
}

fn pixel_dist_m(packed: u32) -> f32 { return f32(packed >> 8u) / 1000.0; }
fn pixel_mat(packed: u32)   -> u32 { return packed & 0xFFu; }

// sigmoid — mesma ordem de operações de `iara_cfc::sigmoid`: `1.0/(1.0+(-x).exp())`.
fn sigmoid_(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wgid: vec3<u32>) {
    let being = wgid.x;
    if being >= params.n_beings { return; }

    // ── 1) ch192-207 usa o ESTADO ANTIGO (h de antes deste tick) ─────────────────
    // Preserva a defasagem de 1 tick que já existia (ver doc do arquivo). Lê ANTES
    // de tocar a retina deste tick.
    let sb = being * CFC_HIDDEN;
    var h: array<f32, 16>;
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        h[i] = cfc_state[sb + i];
    }
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        // Aproximação x/(1+|x|) — CÓPIA LITERAL de perception_temporal.wgsl (bloco
        // "CfC de percepção"). NÃO trocar por tanh() nativo: mudaria o valor do canal.
        let hi = h[i];
        let abs_h = abs(hi);
        let tanh_h = select(hi / (1.0 + abs_h), sign(hi) * 0.999, abs_h > 10.0);
        let val = i32(clamp(tanh_h * 127.0 * 0.3, -127.0, 127.0));
        write_channel(being, 192u + i, val);
    }

    // ── 2) Varre a retina física deste tick: col-pool, min_wall_ahead, edge_row6/7 ──
    var pooled: array<f32, 224>; // 14 rows × 16 bins, row-major (row*16+bin)
    var min_wall_ahead: f32 = NO_WALL_SENTINEL;
    var edge_sum6: i32 = 0;
    var edge_sum7: i32 = 0;
    var edge_prev6: i32 = 0;
    var edge_prev7: i32 = 0;

    for (var row = 0u; row < RETINA_ROWS; row++) {
        let track_edge = row == 6u || row == 7u;
        for (var bin = 0u; bin < POOL_BINS; bin++) {
            let cols_per_bin: u32 = RETINA_COLS / POOL_BINS; // 7
            let lo = bin * cols_per_bin;
            let hi_col: u32 = select(lo + cols_per_bin, RETINA_COLS, bin == POOL_BINS - 1u);
            var mx: i32 = -128;
            for (var col = lo; col < hi_col; col++) {
                let packed = read_retina_pixel(being, row, col);
                let mat    = pixel_mat(packed);
                let dist_m = pixel_dist_m(packed);
                let v = legacy_encoding(dist_m, mat);
                if v > mx { mx = v; }

                if mat == VOXEL_WALL && col >= WALL_COL_LO && col < WALL_COL_HI {
                    min_wall_ahead = min(min_wall_ahead, dist_m);
                }

                // Densidade de borda: soma de |diferença| entre colunas CONSECUTIVAS
                // (0..124 vs 1..125) — espelha context.rs:872-880. Os bins cobrem
                // 0..126 em ordem crescente e sem sobreposição, então percorrer
                // `col` dentro de `bin` crescente reconstrói a sequência 0..125
                // completa; `edge_prev` carrega o valor do pixel anterior através
                // da fronteira entre bins.
                if track_edge {
                    if row == 6u {
                        if col > 0u { edge_sum6 += abs(v - edge_prev6); }
                        edge_prev6 = v;
                    } else {
                        if col > 0u { edge_sum7 += abs(v - edge_prev7); }
                        edge_prev7 = v;
                    }
                }
            }
            pooled[row * POOL_BINS + bin] = f32(mx + 128);
        }
    }

    extra_out[being * 3u + 0u] = min_wall_ahead;
    extra_out[being * 3u + 1u] = clamp(f32(edge_sum6) / 6000.0, 0.0, 1.0);
    extra_out[being * 3u + 2u] = clamp(f32(edge_sum7) / 6000.0, 0.0, 1.0);

    // ── 3) Zone-pool: far=[0,5) mean, mid=[5,10) mean, near=[10,14) max ────────────
    // Espelha `retina_zones()` + `ZoneConfig::pool` — ordem far,mid,near em x[48].
    var x: array<f32, 48>;

    // far: linhas [0,5), MEAN
    for (var c = 0u; c < POOL_BINS; c++) {
        var acc: f32 = 0.0;
        for (var r = 0u; r < 5u; r++) {
            acc += pooled[r * POOL_BINS + c] / 127.5 - 1.0;
        }
        x[c] = acc * (1.0 / 5.0);
    }
    // mid: linhas [5,10), MEAN
    for (var c = 0u; c < POOL_BINS; c++) {
        var acc: f32 = 0.0;
        for (var r = 5u; r < 10u; r++) {
            acc += pooled[r * POOL_BINS + c] / 127.5 - 1.0;
        }
        x[POOL_BINS + c] = acc * (1.0 / 5.0);
    }
    // near: linhas [10,14), MAX
    for (var c = 0u; c < POOL_BINS; c++) {
        var mxv: f32 = -1.0e30;
        for (var r = 10u; r < 14u; r++) {
            let v = pooled[r * POOL_BINS + c] / 127.5 - 1.0;
            if v > mxv { mxv = v; }
        }
        x[2u * POOL_BINS + c] = mxv;
    }

    // ── 4) Passo do CfC — espelha `iara_cfc::CfcCell::step` (ver math/cfc_perception.rs) ──
    // matvec com acumulador sequencial, MESMA ordem de `iara_cfc::matvec`.
    var pre_xi: array<f32, 16>;
    for (var r = 0u; r < CFC_HIDDEN; r++) {
        var acc: f32 = 0.0;
        for (var c = 0u; c < CFC_INPUT; c++) {
            acc += cfc_weights[OFF_W_XI + r * CFC_INPUT + c] * x[c];
        }
        pre_xi[r] = acc;
    }
    var pre_hi: array<f32, 16>;
    for (var r = 0u; r < CFC_HIDDEN; r++) {
        var acc: f32 = 0.0;
        for (var c = 0u; c < CFC_HIDDEN; c++) {
            acc += cfc_weights[OFF_W_HI + r * CFC_HIDDEN + c] * h[c];
        }
        pre_hi[r] = acc;
    }
    var pre_f: array<f32, 16>;
    for (var r = 0u; r < CFC_HIDDEN; r++) {
        var acc: f32 = 0.0;
        for (var c = 0u; c < CFC_HIDDEN; c++) {
            acc += cfc_weights[OFF_W_F + r * CFC_HIDDEN + c] * h[c];
        }
        pre_f[r] = acc;
    }

    var h_new: array<f32, 16>;
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        let b_i = cfc_weights[OFF_B_I + i];
        let b_f = cfc_weights[OFF_B_F + i];
        let i_gate = tanh(pre_xi[i] + pre_hi[i] + b_i);
        let f_gate = sigmoid_(pre_f[i] + b_f);
        let decay  = exp(-f_gate);
        h_new[i] = h[i] * decay + i_gate * (1.0 - decay);
    }

    // ── 5) Persiste h_new para o PRÓXIMO tick ───────────────────────────────────
    for (var i = 0u; i < CFC_HIDDEN; i++) {
        cfc_state[sb + i] = h_new[i];
    }
}
