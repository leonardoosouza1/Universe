// ── FOLHA CORTICAL EQUIPOTENTE — KERNEL B: PLASTICIDADE HEBBIANA ─────────────
//
// Atualiza os pesos de aferência buf_afw via regra Hebb com normalização
// subtrativa (competição por recurso finito):
//
//   Δw_ik = η · (a_i · a_k − α · w_ik · Σ_j a_j · a_k)
//
// A normalização subtrativa faz entradas correlacionadas fortalecerem a conexão
// e entradas descorrelacionadas enfraquecerem — a fronteira emerge onde a
// correlação cruzada cai (doc §4).
//
// Dispatch: (SHEET_N, n_beings, 1) — 1 thread por unidade por ser.
// workgroup_size(32) → cada WG trata 32 unidades de um mesmo ser.

const SHEET_W:  u32 = 32u;
const SHEET_H:  u32 = 32u;
const SHEET_N:  u32 = 1024u;
const N_INPUTS: u32 = 223u;

// Taxa de aprendizado Hebb. Pequena → mudança gradual tick a tick.
const ETA:   f32 = 0.005;
// Coeficiente de normalização subtrativa (competição por recurso finito).
const ALPHA: f32 = 0.003;
// Limites de peso
const W_MAX: f32 =  5.0;
const W_MIN: f32 = -1.0;

struct Params {
    n_beings:   u32,
    tick:       u32,
    // act_sum: Σ_j |act_j| pré-calculado pela CPU e passado aqui para evitar
    // redução intra-kernel que poderia causar race conditions.
    // A CPU lê buf_act após o kernel A e soma os 1024 valores (trivial, 1024 f32).
    act_sum:    f32,
    _pad0:      u32,
}

@group(0) @binding(0) var<uniform>             params:   Params;
// Ativação da folha (saída do kernel A). Layout: ser*SHEET_N + unidade.
@group(0) @binding(1) var<storage, read>       buf_act:  array<f32>;
// Sinal sensorial. Layout: ser*N_INPUTS + canal.
@group(0) @binding(2) var<storage, read>       buf_wave: array<i32>;
// Pesos de aferência hebbiana. Layout: ser*SHEET_N*N_INPUTS + unidade*N_INPUTS + canal.
@group(0) @binding(3) var<storage, read_write> buf_afw:  array<f32>;

// ── Kernel B: 1 thread por unidade, dispatch (SHEET_N/32, n_beings, 1) ────────

@compute @workgroup_size(32, 1, 1)
fn hebb(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let unit:  u32 = gid.x;   // índice na folha [0, SHEET_N)
    let being: u32 = gid.y;   // índice do ser

    if unit >= SHEET_N || being >= params.n_beings { return; }

    let a_i:       f32 = buf_act[being * SHEET_N + unit]; // ativação unidade i
    let wave_base: u32 = being * N_INPUTS;
    let afw_row:   u32 = being * SHEET_N * N_INPUTS + unit * N_INPUTS;
    let act_sum:   f32 = params.act_sum; // Σ_j |act_j| pré-calculado pela CPU

    // Δw_ik = η · (a_i · a_k − α · w_ik · |a_k| · act_sum)
    for (var k: u32 = 0u; k < N_INPUTS; k = k + 1u) {
        let a_k:    f32 = f32(buf_wave[wave_base + k]) / 127.0;
        let w_old:  f32 = buf_afw[afw_row + k];

        let hebb_term: f32 = a_i * a_k;
        // Normalização subtrativa: proporcional ao peso atual e à atividade total
        let norm_term: f32 = ALPHA * w_old * act_sum * abs(a_k);
        let delta_w:   f32 = ETA * (hebb_term - norm_term);

        buf_afw[afw_row + k] = clamp(w_old + delta_w, W_MIN, W_MAX);
    }
}
